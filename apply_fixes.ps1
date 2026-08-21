$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Applying RAG refactor fixes to: $ProjectRoot"

$files = @(
    "backend/core/config.py",
    "backend/core/llm.py",
    "backend/services/document_service.py",
    "backend/retrieval/vector_store.py",
    "backend/retrieval/retriever.py",
    "backend/retrieval/reranker.py",
    "backend/chains/retrieval.py",
    "backend/chains/rag_chain.py",
    "backend/tests/test_rag_chain.py"
)

foreach ($relative in $files) {
    $source = Join-Path $PatchRoot $relative
    $destination = Join-Path $ProjectRoot $relative

    if (-not (Test-Path $source)) {
        throw "Patch file missing: $source"
    }

    $parent = Split-Path $destination -Parent
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    Copy-Item $source $destination -Force
    Write-Host "Updated $relative"
}

Write-Host ""
Write-Host "Checking model settings..."
python -c "from backend.core.config import get_settings; s=get_settings(); print('generation_model =', s.generation_model); print('router_model =', s.router_model)"

Write-Host ""
Write-Host "IMPORTANT: rebuild your FAISS index after this because chunk IDs are generated during ingestion."
Write-Host "Then run: uv run pytest -v"
