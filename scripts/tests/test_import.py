#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from backend.api.main import app
    print('✅ Full backend app imported successfully')
    print('Routes available:')
    count = 0
    for route in app.routes:
        if hasattr(route, 'path'):
            if 'reconstruct' in route.path:
                print(f'  {route.methods} {route.path}')
                count += 1
    print(f'Total routes with reconstruct: {count}')
except Exception as e:
    print(f'❌ Failed to import full backend: {e}')
    import traceback
    traceback.print_exc()