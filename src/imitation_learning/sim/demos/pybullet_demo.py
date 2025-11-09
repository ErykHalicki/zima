import pybullet as p
import time

p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

p.setGravity(0,0,-10)

planeId = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_PLANE)
)
boxId = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5,0.5,0.5]),
    basePosition=[0,0,1]
)

pixelWidth = 640
pixelHeight = 480
camTargetPos = [0, 0, 0]
camEyePos = [0, 0, 2.5]
camUpVec = [0, 1, 0]

projectionMatrix = p.computeProjectionMatrixFOV(90, float(pixelWidth)/pixelHeight, 0.001, 100.0)

view_matrix = p.computeViewMatrix([2, 0, 1], [0, 0, 0], [0, 0, 1])

camera_fps = 10  
physics_fps = 100
steps_per_capture = physics_fps / camera_fps

p.setTimeStep(1./physics_fps)
p.setRealTimeSimulation(1)

# FPS calculation variables
step_num = 0
frame_count = 0
fps_update_interval = 10  # Update FPS display every N frames
start_time = time.time()
last_fps_time = start_time

while p.isConnected():
    step_num += 1
    p.stepSimulation()
    # ----------------
    if step_num % steps_per_capture == 0:
        frame_count += 1

        # Calculate and display FPS
        if frame_count % fps_update_interval == 0:
            current_time = time.time()
            elapsed = current_time - last_fps_time
            fps = fps_update_interval / elapsed
            print(f"FPS: {fps:.2f}")
            last_fps_time = current_time

        viewMatrix = p.computeViewMatrix(camEyePos, camTargetPos, camUpVec)
        img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=0,
                               lightDirection=[1, 1, 1])

p.disconnect()
