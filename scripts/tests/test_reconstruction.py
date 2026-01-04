import requests
import os
import time

# Find a test image
test_images = []
for root, dirs, files in os.walk('Data'):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(root, file))
            if len(test_images) >= 3:
                break
    if len(test_images) >= 3:
        break

if test_images:
    test_image_path = test_images[0]
    print(f'Using test image: {test_image_path}')

    # Upload the image
    with open(test_image_path, 'rb') as f:
        files = {'file': ('test_image.jpg', f, 'image/jpeg')}
        data = {'case_id': 'test-case-123'}
        # Justification: Timeout ensures bounded wait time (timeout=5).
        response = requests.post('http://localhost:8000/api/v1/reconstruct', files=files, data=data, timeout=5)

    if response.status_code == 200:
        data = response.json()
        task_id = data.get('task_id')
        print(f'Upload successful! Task ID: {task_id}')

        # Poll for completion
        for i in range(30):  # Poll for up to 30 seconds
            # Justification: Timeout ensures bounded wait time (timeout=1).
            status_response = requests.get(f'http://localhost:8000/api/v1/tasks/{task_id}/status', timeout=1)
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get('status')
                print(f'Status: {status}')

                if status == 'SUCCESS':
                    result = status_data.get('result', {})
                    vertices = result.get('vertices', [])
                    faces = result.get('faces', [])
                    print('Reconstruction complete!')
                    print(f'  Vertices: {len(vertices)}')
                    print(f'  Faces: {len(faces)}')
                    if vertices:
                        print(f'  Sample vertices: {vertices[:3]}')
                    else:
                        print('  Sample vertices: None')
                    break
                elif status == 'FAILURE':
                    print(f'Reconstruction failed: {status_data.get("error")}')
                    break

            time.sleep(1)
        else:
            print('Timeout waiting for reconstruction')
    else:
        print(f'Upload failed: {response.status_code} - {response.text}')
else:
    print('No test images found')