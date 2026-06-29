"""
Deploy script — Technical Documentation Assistant en LocalStack.

Requisitos:
  - LocalStack corriendo (community o pro)
  - ENVIRONMENT=localstack en .env
  - OPENAI_API_KEY en .env

El script es idempotente: puede re-ejecutarse sin efectos secundarios.
Gestiona automáticamente:
  1. Límite de tamaño de Lambda (via config API de LocalStack Pro)
  2. Índice ChromaDB en S3 (para que la Lambda tenga el RAG disponible)
  3. Paquete Lambda con wheels Linux (compatible con el runtime del contenedor)
  4. API Gateway → Lambda (proxy completo)
"""

import sys
import os
import stat
import zipfile
import subprocess
import tempfile
import shutil
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import boto3
from botocore.exceptions import ClientError
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ------------------------------------------------------------------ #
#  Constantes                                                          #
# ------------------------------------------------------------------ #

LAMBDA_FUNCTION_NAME = "technical-docs-assistant"
LAMBDA_HANDLER       = "src.api.main.handler"
LAMBDA_RUNTIME       = "python3.11"
LAMBDA_TIMEOUT       = 60    # segundos — cold start con embeddings
LAMBDA_MEMORY        = 512   # MB
API_GATEWAY_NAME     = "docs-assistant-api"
LAMBDA_ROLE_ARN      = "arn:aws:iam::000000000000:role/lambda-role"
LAMBDA_S3_KEY        = "lambda/function.zip"
S3_CHROMA_KEY        = "chroma_db/index.tar.gz"

# Tamaño máximo descomprimido que queremos configurar en LocalStack (1 GB).
# El paquete con dependencias Python modernas supera los 250 MB predeterminados.
_LOCALSTACK_LAMBDA_LIMIT = 1_073_741_824


# ------------------------------------------------------------------ #
#  Helper boto3                                                        #
# ------------------------------------------------------------------ #

def _boto3_client(service: str, settings):
    kwargs = {"service_name": service, "region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client(**kwargs)


# ------------------------------------------------------------------ #
#  PASO 0 — Expandir límite de Lambda en LocalStack                   #
# ------------------------------------------------------------------ #

def expand_lambda_limit(endpoint_url: str) -> bool:
    """
    Amplía LAMBDA_LIMITS_TOTAL_CODE_SIZE en LocalStack vía su config API.
    Disponible en LocalStack Pro; en Community el límite se debe pasar como
    env var al iniciar el contenedor.

    Devuelve True si el límite quedó configurado correctamente.
    """
    import requests as _req

    target = _LOCALSTACK_LAMBDA_LIMIT
    config_url = f"{endpoint_url}/_localstack/config"

    # Intentar actualizar (LocalStack Pro acepta PATCH o POST)
    for method in ("patch", "post"):
        try:
            fn = getattr(_req, method)
            resp = fn(config_url, json={"LAMBDA_LIMITS_TOTAL_CODE_SIZE": target}, timeout=5)
            if resp.status_code in (200, 204):
                break
        except Exception:
            pass

    # Verificar leyendo la configuración actual
    try:
        r = _req.get(config_url, timeout=5)
        if r.status_code == 200:
            current = r.json().get("LAMBDA_LIMITS_TOTAL_CODE_SIZE")
            if current and int(current) >= target:
                logger.info(
                    f"LocalStack Lambda limit: {int(current) // 1024 // 1024} MB ✓"
                )
                return True
    except Exception:
        pass

    logger.warning(
        "No se pudo actualizar el límite de Lambda via config API "
        "(normal en LocalStack Community).\n"
        "Si el deploy falla por tamaño, reiniciar LocalStack con:\n\n"
        f"  docker run --rm -d --name localstack-main \\\n"
        f"    -p 127.0.0.1:4566:4566 \\\n"
        f"    -e LAMBDA_LIMITS_TOTAL_CODE_SIZE={target} \\\n"
        f"    localstack/localstack-pro\n"
    )
    return False


# ------------------------------------------------------------------ #
#  PASO 1 — Paquete Lambda                                            #
# ------------------------------------------------------------------ #

# Extensiones de binarios compilados para Windows — la Lambda corre en Linux.
_ZIP_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyi", ".pyx", ".pyd", ".dll"}

_ZIP_EXCLUDE_DIRS = {
    "__pycache__", "tests", "test", "testing",
    "doc", "docs", "examples", "benchmarks",
    "bin", "Scripts", "share", "htmlcov",
    "locale", "locales", "i18n",
}

# Paquetes top-level que no se necesitan en runtime Lambda.
# Objetivo: mantener el ZIP descomprimido dentro del límite de Lambda.
_ZIP_EXCLUDE_TOP_LEVEL = {
    # onnxruntime: modelo de embedding por defecto de ChromaDB (~150 MB).
    # No se usa porque empleamos OpenAIEmbeddings via API.
    "onnxruntime", "onnxruntime_common",
    # Ecosistema Hugging Face — irrelevante con OpenAI Embeddings
    "datasets", "huggingface_hub", "tokenizers",
    # Deps transitivas grandes de datasets (no usadas en runtime)
    "pyarrow", "multiprocess", "dill", "xxhash",
    # gRPC — ChromaDB embedded (PersistentClient) no lo requiere.
    # NOTA: el paquete pip "grpcio" se instala como directorio "grpc" en Linux.
    "grpcio", "grpc",
    "grpcio_tools", "grpc_tools",
    # Telemetría de ChromaDB — no necesaria en producción
    "posthog",
    # Extras de uvicorn[standard] — innecesarios en Lambda (Mangum gestiona el ciclo)
    "uvloop", "watchfiles", "httptools",
}

# Servicios de AWS que la Lambda usa en runtime.
# botocore/data/ tiene modelos JSON para 400+ servicios (~40-50 MB descomprimido);
# excluir los no usados es el mayor ahorro de espacio disponible.
_BOTOCORE_KEEP_SERVICES = {"s3"}


def _should_exclude(file_path: Path, tmp_dir: Path) -> bool:
    rel_parts = file_path.relative_to(tmp_dir).parts

    if rel_parts and rel_parts[0] in _ZIP_EXCLUDE_TOP_LEVEL:
        return True
    if file_path.suffix in _ZIP_EXCLUDE_SUFFIXES:
        return True
    for part in rel_parts:
        if part in _ZIP_EXCLUDE_DIRS:
            return True
        if part.endswith(".dist-info") or part.endswith(".egg-info"):
            return True
    # Modelos de servicio botocore no utilizados por la Lambda
    if (
        len(rel_parts) >= 3
        and rel_parts[0] == "botocore"
        and rel_parts[1] == "data"
        and rel_parts[2] not in _BOTOCORE_KEEP_SERVICES
    ):
        return True
    return False


def build_lambda_package() -> Path:
    """
    Genera function.zip con wheels Linux (manylinux2014_x86_64).
    Lambda corre en Linux dentro del contenedor Docker de LocalStack,
    por lo que los binarios deben ser .so (no .pyd/.dll de Windows).
    pip >= 22 soporta --platform para cross-compilation sin necesitar Linux.
    """
    logger.info("Building Lambda deployment package...")

    tmp_dir = Path(tempfile.mkdtemp()) / "lambda_package"
    tmp_dir.mkdir(parents=True)

    reqs = root_path / "requirements-lambda.txt"
    if not reqs.exists():
        reqs = root_path / "requirements.txt"

    logger.info(f"Installing Linux wheels (manylinux2014_x86_64) from {reqs.name}...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--platform", "manylinux2014_x86_64",
            "--implementation", "cp",
            "--python-version", "311",
            "--only-binary", ":all:",
            "-r", str(reqs),
            "-t", str(tmp_dir),
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Algunos paquetes no tienen wheel manylinux — instalar del OS actual como fallback.
        # La Lambda solo funcionará si LocalStack usa LAMBDA_EXECUTOR=local.
        logger.warning(
            "Cross-platform install failed (some packages lack manylinux wheel). "
            "Falling back to local OS packages — Lambda may fail at runtime.\n"
            + result.stderr[:400]
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(reqs), "-t", str(tmp_dir), "-q"],
            check=True,
        )
    elif result.stderr.strip():
        logger.warning(result.stderr.strip()[:400])

    # Copiar código fuente del proyecto
    src_dst = tmp_dir / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(root_path / "src", src_dst)

    # Comprimir, excluyendo binarios Windows y paquetes innecesarios
    zip_path = root_path / "function.zip"
    included = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in tmp_dir.rglob("*"):
            if fp.is_file() and not _should_exclude(fp, tmp_dir):
                zf.write(fp, fp.relative_to(tmp_dir))
                included += 1

    def _rm_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(tmp_dir, onerror=_rm_readonly)
    size_mb = zip_path.stat().st_size / 1024 / 1024
    logger.info(f"Package built: {zip_path.name} ({size_mb:.1f} MB, {included} files)")
    return zip_path


# ------------------------------------------------------------------ #
#  PASO 2 — S3: bucket, documentos y ChromaDB                         #
# ------------------------------------------------------------------ #

def setup_s3(s3_client, bucket_name: str, docs_dir: Path):
    """Crea el bucket y sube los documentos Markdown."""
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"S3 bucket '{bucket_name}' created.")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
        logger.info(f"S3 bucket '{bucket_name}' already exists.")

    if docs_dir.exists():
        uploaded = 0
        for doc in docs_dir.rglob("*.md"):
            key = "documents/" + doc.relative_to(docs_dir).as_posix()
            s3_client.upload_file(str(doc), bucket_name, key)
            uploaded += 1
        logger.info(f"Uploaded {uploaded} documents to s3://{bucket_name}/documents/")
    else:
        logger.warning(f"Documents directory not found at {docs_dir} — skipping upload.")


def ensure_chroma_in_s3(s3_client, bucket_name: str):
    """
    Verifica que el índice ChromaDB esté en S3.
    Si no existe, ejecuta la indexación local y lo sube.
    La Lambda necesita este archivo para tener el RAG disponible en cold start.
    """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=S3_CHROMA_KEY)
        logger.info(f"ChromaDB index found in S3 (s3://{bucket_name}/{S3_CHROMA_KEY}).")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise

    logger.info("ChromaDB index not found in S3. Running indexing...")
    from scripts.index_documents import index_documents, upload_index_to_s3, LOCAL_CHROMA_DIR
    index_documents(LOCAL_CHROMA_DIR)
    upload_index_to_s3(LOCAL_CHROMA_DIR)


# ------------------------------------------------------------------ #
#  PASO 3 — Lambda                                                    #
# ------------------------------------------------------------------ #

def deploy_lambda(lambda_client, s3_client, zip_path: Path, bucket_name: str, env_vars: dict):
    """Sube el ZIP a S3 y crea/actualiza la función Lambda."""
    size_mb = zip_path.stat().st_size / 1024 / 1024
    logger.info(f"Uploading {zip_path.name} ({size_mb:.1f} MB) to s3://{bucket_name}/{LAMBDA_S3_KEY}")
    s3_client.upload_file(str(zip_path), bucket_name, LAMBDA_S3_KEY)
    logger.info("Lambda ZIP uploaded to S3.")

    code_ref = {"S3Bucket": bucket_name, "S3Key": LAMBDA_S3_KEY}

    try:
        lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime=LAMBDA_RUNTIME,
            Role=LAMBDA_ROLE_ARN,
            Handler=LAMBDA_HANDLER,
            Code=code_ref,
            Timeout=LAMBDA_TIMEOUT,
            MemorySize=LAMBDA_MEMORY,
            Environment={"Variables": env_vars},
        )
        logger.info(f"Lambda '{LAMBDA_FUNCTION_NAME}' created.")

    except ClientError as e:
        code = e.response["Error"]["Code"]

        if code == "ResourceConflictException":
            logger.info(f"Lambda '{LAMBDA_FUNCTION_NAME}' exists — updating.")
            lambda_client.update_function_code(
                FunctionName=LAMBDA_FUNCTION_NAME,
                S3Bucket=bucket_name,
                S3Key=LAMBDA_S3_KEY,
            )
            lambda_client.update_function_configuration(
                FunctionName=LAMBDA_FUNCTION_NAME,
                Timeout=LAMBDA_TIMEOUT,
                MemorySize=LAMBDA_MEMORY,
                Environment={"Variables": env_vars},
            )
            logger.info(f"Lambda '{LAMBDA_FUNCTION_NAME}' updated.")

        elif code == "InvalidParameterValueException" and "Unzipped size" in str(e):
            logger.error(
                "\n" + "=" * 60 + "\n"
                "ERROR: El paquete Lambda supera el límite de 250 MB descomprimido.\n\n"
                "Las dependencias Python modernas (chromadb, langchain, pydantic, numpy)\n"
                "superan el límite predeterminado de Lambda/LocalStack.\n\n"
                "SOLUCIÓN — Reiniciar LocalStack con límite ampliado a 1 GB:\n\n"
                "  # Detener el container actual:\n"
                "  docker stop localstack-main\n\n"
                "  # Iniciar con límite ampliado:\n"
                "  docker run --rm -d --name localstack-main \\\n"
                "    -p 127.0.0.1:443:443 \\\n"
                "    -p 127.0.0.1:4510-4560:4510-4560 \\\n"
                "    -p 127.0.0.1:4566:4566 \\\n"
                f"    -e LAMBDA_LIMITS_TOTAL_CODE_SIZE={_LOCALSTACK_LAMBDA_LIMIT} \\\n"
                "    localstack/localstack-pro\n\n"
                "  # Luego re-ejecutar este script.\n"
                + "=" * 60
            )
            raise
        else:
            raise


# ------------------------------------------------------------------ #
#  PASO 4 — API Gateway                                               #
# ------------------------------------------------------------------ #

def _put_method_safe(apigw_client, **kwargs):
    """put_method ignorando ConflictException (idempotente)."""
    try:
        apigw_client.put_method(**kwargs)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConflictException":
            raise


def setup_api_gateway(apigw_client, lambda_client, settings) -> str:
    """Crea o reutiliza una REST API Gateway con proxy → Lambda."""
    apis = apigw_client.get_rest_apis().get("items", [])
    api = next((a for a in apis if a["name"] == API_GATEWAY_NAME), None)

    if api:
        api_id = api["id"]
        logger.info(f"API Gateway '{API_GATEWAY_NAME}' already exists (id={api_id}).")
    else:
        api = apigw_client.create_rest_api(name=API_GATEWAY_NAME)
        api_id = api["id"]
        logger.info(f"API Gateway '{API_GATEWAY_NAME}' created (id={api_id}).")

    resources = apigw_client.get_resources(restApiId=api_id)["items"]
    root_res = next(r for r in resources if r["path"] == "/")

    proxy_res = next((r for r in resources if r.get("pathPart") == "{proxy+}"), None)
    if not proxy_res:
        proxy_res = apigw_client.create_resource(
            restApiId=api_id, parentId=root_res["id"], pathPart="{proxy+}"
        )

    lambda_arn = lambda_client.get_function(
        FunctionName=LAMBDA_FUNCTION_NAME
    )["Configuration"]["FunctionArn"]

    uri = (
        f"arn:aws:apigateway:{settings.aws_region}:lambda:path/2015-03-31"
        f"/functions/{lambda_arn}/invocations"
    )

    # Proxy resource {proxy+}
    _put_method_safe(
        apigw_client,
        restApiId=api_id, resourceId=proxy_res["id"],
        httpMethod="ANY", authorizationType="NONE",
    )
    apigw_client.put_integration(
        restApiId=api_id, resourceId=proxy_res["id"],
        httpMethod="ANY", type="AWS_PROXY",
        integrationHttpMethod="POST", uri=uri,
    )

    # Recurso raíz "/" (evita 403 en requests sin path)
    _put_method_safe(
        apigw_client,
        restApiId=api_id, resourceId=root_res["id"],
        httpMethod="ANY", authorizationType="NONE",
    )
    apigw_client.put_integration(
        restApiId=api_id, resourceId=root_res["id"],
        httpMethod="ANY", type="AWS_PROXY",
        integrationHttpMethod="POST", uri=uri,
    )

    # Permiso para que API Gateway invoque la Lambda
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId="apigw-invoke",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{settings.aws_region}:000000000000:{api_id}/*/*",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ResourceConflictException",):
            raise

    apigw_client.create_deployment(restApiId=api_id, stageName="dev")
    endpoint = f"{settings.aws_endpoint_url}/restapis/{api_id}/dev/_user_request_"
    logger.info(f"API Gateway deployed. Base URL: {endpoint}")
    return endpoint


# ------------------------------------------------------------------ #
#  PASO 5 — Verificación post-deploy                                  #
# ------------------------------------------------------------------ #

def verify_deployment(base_url: str):
    """
    Verifica el deploy invocando /health y /test-rag.
    El primer request tiene cold start: Lambda descarga ChromaDB de S3 (~segundos).
    """
    import requests as _req
    import time

    logger.info("Running post-deploy verification...")

    # /health — puede necesitar un par de reintentos por cold start
    health_ok = False
    for attempt in range(1, 4):
        try:
            r = _req.get(f"{base_url}/health", timeout=30)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "unknown")
                components = data.get("components", {})
                logger.info(f"Health check: {status} — {components}")
                if status == "healthy":
                    health_ok = True
                elif status == "degraded":
                    logger.warning(
                        "ChromaDB reporta error en /health. "
                        "Si el índice no estaba en S3, ejecutar:\n"
                        "  python scripts/index_documents.py --upload\n"
                        "y luego reinvocar el endpoint."
                    )
                break
            else:
                logger.warning(f"Health check HTTP {r.status_code} (intento {attempt}/3)")
        except Exception as exc:
            if attempt < 3:
                logger.info(f"Cold start en curso, reintentando ({attempt}/3)... {exc}")
                time.sleep(5)
            else:
                logger.warning(f"Health check timeout: {exc}")

    # /test-rag — valida que el RAG devuelve resultados reales
    try:
        r = _req.get(f"{base_url}/test-rag", params={"query": "FastAPI"}, timeout=45)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                logger.info(f"RAG test OK — {len(results)} result(s), first source: {results[0].get('source')}")
            else:
                logger.warning(
                    "RAG test returned 0 results. El índice ChromaDB puede estar vacío.\n"
                    "Ejecutar: python scripts/index_documents.py --upload"
                )
        else:
            logger.warning(f"RAG test HTTP {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"RAG test failed: {exc}")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    settings = get_settings()

    if settings.environment != "localstack":
        logger.error(
            "ENVIRONMENT must be 'localstack'. Set it in your .env file:\n"
            "  ENVIRONMENT=localstack"
        )
        sys.exit(1)

    logger.info(f"Deploying to LocalStack at {settings.aws_endpoint_url}")

    # 0. Ampliar límite de Lambda (LocalStack Pro: via config API)
    expand_lambda_limit(settings.aws_endpoint_url)

    s3_client     = _boto3_client("s3", settings)
    lambda_client = _boto3_client("lambda", settings)
    apigw_client  = _boto3_client("apigateway", settings)

    env_vars = {
        "ENVIRONMENT":      "localstack",
        "AWS_ENDPOINT_URL": settings.aws_endpoint_url,
        "AWS_REGION":       settings.aws_region,
        "S3_BUCKET_NAME":   settings.s3_bucket_name,
        "OPENAI_API_KEY":   settings.openai_api_key,
    }
    if settings.openweather_api_key:
        env_vars["OPENWEATHER_API_KEY"] = settings.openweather_api_key

    # 1. S3: bucket + documentos Markdown
    setup_s3(s3_client, settings.s3_bucket_name, root_path / "data" / "documents")

    # 2. S3: ChromaDB index (necesario para el RAG en Lambda)
    ensure_chroma_in_s3(s3_client, settings.s3_bucket_name)

    # 3. Lambda: compilar y desplegar
    zip_path = build_lambda_package()
    deploy_lambda(lambda_client, s3_client, zip_path, settings.s3_bucket_name, env_vars)

    # 4. API Gateway
    base_url = setup_api_gateway(apigw_client, lambda_client, settings)

    # 5. Verificación
    verify_deployment(base_url)

    logger.info("=" * 60)
    logger.info("Deployment complete!")
    logger.info(f"Base URL: {base_url}")
    logger.info("")
    logger.info("Quick tests:")
    logger.info(f'  curl "{base_url}/health"')
    logger.info(f'  curl "{base_url}/test-rag?query=FastAPI"')
    logger.info(
        f'  curl -X POST "{base_url}/query" '
        f'-H "Content-Type: application/json" '
        f'-d \'{{"query":"¿Cómo crear un endpoint en FastAPI?"}}\''
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
