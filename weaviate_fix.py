import asyncio
import logging
import inspect
from typing import Any, Coroutine, Callable, Optional

logger = logging.getLogger(__name__)

def run_async(coro_or_func, *args, **kwargs):
    """
    Helper function to run async coroutines in a synchronous context.
    Also handles non-coroutines gracefully for Weaviate v4 compatibility.
    
    Args:
        coro_or_func: The coroutine, coroutine function, or regular function to execute
        *args, **kwargs: Arguments to pass if coro_or_func is a function
        
    Returns:
        The result of the coroutine or function
    """
    # Check if it's already a coroutine
    if asyncio.iscoroutine(coro_or_func):
        coro = coro_or_func
    # Check if it's a coroutine function that needs to be called
    elif asyncio.iscoroutinefunction(coro_or_func):
        coro = coro_or_func(*args, **kwargs)
    # If it's a regular function or object, just return it directly
    else:
        # If it's callable but not a coroutine function, call it
        if callable(coro_or_func) and not isinstance(coro_or_func, (str, list, dict, tuple)):
            try:
                return coro_or_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error calling function: {e}")
                raise
        # If it's not callable, just return it
        return coro_or_func
    
    # Handle the coroutine
    try:
        # Python 3.12 and newer: new_event_loop() is preferred over get_event_loop()
        try:
            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        logger.error(f"Error running async code: {e}")
        raise

def is_coroutine(obj):
    """
    Check if an object is a coroutine, coroutine function, or awaitable.
    Handles both functions and objects for Weaviate v4 compatibility.
    """
    if obj is None:
        return False
        
    # Direct coroutine check
    if asyncio.iscoroutine(obj):
        return True
        
    # Coroutine function check
    if asyncio.iscoroutinefunction(obj):
        return True
        
    # Check if it's awaitable
    if hasattr(obj, "__await__"):
        return True
        
    # For method objects, check the underlying function
    if hasattr(obj, "__func__"):
        return asyncio.iscoroutinefunction(obj.__func__)
        
    return False

def safe_run(func, *args, **kwargs):
    """
    Safely run a function that might be coroutine or regular function.
    This is particularly useful for Weaviate client v4 methods.
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Result of the function
    """
    if is_coroutine(func):
        return run_async(func, *args, **kwargs)
    elif callable(func):
        return func(*args, **kwargs)
    else:
        return func