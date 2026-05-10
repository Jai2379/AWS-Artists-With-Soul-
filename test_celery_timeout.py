import time
from celery import Celery

REDIS_URL = "redis://localhost:6379/0"
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task
def dummy_task():
    pass

print("Attempting to connect to Redis via Celery delay()...")
start = time.time()
try:
    # Set a short timeout for the connection
    dummy_task.delay()
except Exception as e:
    print(f"Exception caught: {type(e).__name__}: {e}")

print(f"Time taken to fail/succeed: {time.time() - start:.2f} seconds")
