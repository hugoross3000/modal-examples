# # Run multiple instances of a model on a single GPU

# Many models are small enough to fit multiple instances onto a single GPU.
# Doing so can dramatically reduce the number of GPUs needed to handle demand.

# We use `allow_concurrent_inputs` to allow multiple connections into the container
# We load the model instances into a FIFO queue to ensure only one http handler can access it at once

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "sentence-transformers==3.2.0"
)

app = modal.App("example-gpu-packing", image=image)


class ModelPool:
    # ModelPool holds multiple instances of the model, using a queue
    def __init__(self):
        self.pool: asyncio.Queue = asyncio.Queue()

    async def put(self, model):
        await self.pool.put(model)

    # We provide a context manager to easily acquire and release models from the pool
    @asynccontextmanager
    async def acquire_model(self):
        model = await self.pool.get()
        try:
            yield model
        finally:
            await self.pool.put(model)


with image.imports():
    from sentence_transformers import SentenceTransformer


@app.cls(
    gpu="A10G",
    concurrency_limit=1,  # max one container for this app, for the sake of demoing concurrent_inputs
    allow_concurrent_inputs=1000,  # allow multiple inputs to be concurrently processed by one container
)
class Server:
    def __init__(self, n_models=10):
        self.model_pool = ModelPool()
        self.n_models = n_models

    @modal.build()
    def download(self):
        model = SentenceTransformer("BAAI/bge-m3")
        model.save("/model.bge")

    @modal.enter()
    async def load_models(self):
        # Boot N models onto the gpu, and place into the pool
        t0 = time.time()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.n_models)

        async def run_in_thread_pool(func, *args):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.thread_pool, func, *args)

        self.run_in_thread_pool = run_in_thread_pool

        for i in range(self.n_models):
            model = SentenceTransformer("/model.bge", device="cuda")
            await self.model_pool.put(model)

        print(f"📦 Loading {self.n_models} models took {time.time() - t0:.4f}s")

    @modal.method()
    def prewarm(self):
        pass

    @modal.method()
    async def predict(self, sentence):
        # Block until a model is available
        async with self.model_pool.acquire_model() as model:
            # We now have exclusive access to this model instance,
            # and so we can put it into a thread and await the result
            embedding = await self.run_in_thread_pool(model.encode, sentence)
        return embedding.tolist()

    @modal.exit()
    def exit(self):
        self.thread_pool.shutdown()


@app.local_entrypoint()
async def main(n_requests: int = 100):
    # We benchmark with 100 requests in parallel.
    # Thanks to allow_concurrent_inputs, 100 requests will enter .predict() at the same time.

    bold, end = "\033[1m", "\033[0m"
    print(f"{bold}📦 Testing with {n_requests} concurrent requests{end}\n")
    sentences = [f"Sentence {ii} " * 25 for ii in range(n_requests)]

    # Baseline: a server with a pool size of 1 model
    print(f"{bold}📦 Testing Baseline (1 Model){end}")
    t0 = time.time()
    server = Server(n_models=1)
    server.prewarm.remote()
    print("📦 Container boot took {:.4f}s".format(time.time() - t0))

    t0 = time.time()
    async for result in server.predict.map.aio(sentences, order_outputs=False):
        pass
    print(f"📦 Inference took {time.time() - t0:.4f}s\n")

    # Packing: a server with a pool size of 8 models
    # Note: this increases boot time, but reduces inference time
    print(f"{bold}📦 Testing Packing (8 Models){end}")
    t0 = time.time()
    server = Server(n_models=8)
    server.prewarm.remote()
    print("📦 Container boot took {:.4f}s".format(time.time() - t0))

    t0 = time.time()
    async for result in server.predict.map.aio(sentences, order_outputs=False):
        pass
    print(f"📦 Inference took {time.time() - t0:.4f}s\n")
