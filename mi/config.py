import os
import pathlib
from mi.utils.env_utils import OpenAIKeyRing, load_oai_keys

# Initialize directories
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR = ROOT_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = ROOT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Load API keys and create OpenAIKeyRing
oai_key_ring = OpenAIKeyRing(load_oai_keys(ROOT_DIR))

# Concurrency limits for LLM API calls
# Modal uses semaphores to limit concurrent requests to prevent 429 rate limit errors
# Higher values = more parallel requests but increased GPU/queue pressure on vLLM
MODAL_SAMPLE_CONCURRENCY = int(os.environ.get('MODAL_SAMPLE_CONCURRENCY', '50'))

# OpenAI concurrency limit for sampling and judging
# OpenAI can handle 1000+ concurrent requests, limited by token throughput not connections
OPENAI_SAMPLE_CONCURRENCY = int(os.environ.get('OPENAI_SAMPLE_CONCURRENCY', '1000'))