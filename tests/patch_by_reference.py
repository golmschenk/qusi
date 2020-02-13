from unittest.mock import patch

def patch_reference():
    def decorated_function(reference):
        patch(f'{reference.__module__}.{reference.__name__}')

    return decorated_function
