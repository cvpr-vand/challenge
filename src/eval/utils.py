import functools
import torch

def ispow2(n):
    return n > 0 and (n & (n - 1)) == 0

def auto_batch_size(max_batch_size=128, min_batch_size=1):
    """
    Decorator that retries the function with decreasing powers of two as batch size
    on OOM errors. Validates that starting_batch_size is a power of two.
    """
    if not ispow2(max_batch_size):
        raise ValueError(f"starting_batch_size must be a power of 2, got {max_batch_size}")

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs.get('batch_size') is not None:
                # when passed explicitly, we don't auto-configure the batch size
                print("\nUsing pre-specified batch size:", kwargs['batch_size'])
                return fn(*args, **kwargs), kwargs['batch_size']
            batch_size = max_batch_size
            while batch_size >= min_batch_size:
                try:
                    print(f"\nTrying with batch size {batch_size}")
                    torch.cuda.empty_cache()
                    kwargs["batch_size"] = batch_size
                    result = fn(*args, **kwargs)
                    print(f"\nSuccess with batch size: {batch_size}")
                    return result, batch_size
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"\nOOM at batch size {batch_size}, trying smaller...")
                        torch.cuda.empty_cache()
                        batch_size //= 2
                    else:
                        raise e
            raise RuntimeError(f"\nAll batch sizes down to {min_batch_size} caused OOM.")
        return wrapper
    return decorator
