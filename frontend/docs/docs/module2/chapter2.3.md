---
sidebar_position: 4
---

# Chapter 2.3: Sensory Input: Simulating LiDAR, Depth Cameras, and IMUs

## Introduction to Sensor Simulation

In robotics, accurate sensor simulation is crucial for developing and testing perception algorithms. This chapter covers simulating common robot sensors: LiDAR, depth cameras, and IMUs in simulation environments.

## LiDAR Simulation

### Raycasting-Based LiDAR

LiDAR (Light Detection and Ranging) sensors work by emitting laser beams and measuring the time it takes for reflections to return. In simulation, this is typically implemented using raycasting:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SimulatedLiDAR : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int horizontalRays = 360;  // Number of horizontal rays
    public int verticalRays = 16;     // Number of vertical rays (for 3D LiDAR)
    public float minAngle = -90f;     // Minimum horizontal angle
    public float maxAngle = 90f;      // Maximum horizontal angle
    public float minVerticalAngle = -15f;  // Minimum vertical angle
    public float maxVerticalAngle = 15f;   // Maximum vertical angle
    public float maxRange = 30.0f;    // Maximum detection range
    public LayerMask detectionMask = -1;   // Layers to detect
    public float updateRate = 10f;    // Update rate in Hz

    [Header("Visualization")]
    public bool visualizeRays = true;
    public float rayVisualizationTime = 0.02f;

    private float nextUpdate = 0f;
    private List<float> ranges;

    void Start()
    {
        ranges = new List<float>(new float[horizontalRays * verticalRays]);
    }

    void Update()
    {
        if (Time.time >= nextUpdate)
        {
            SimulateLiDAR();
            nextUpdate = Time.time + (1f / updateRate);
        }
    }

    void SimulateLiDAR()
    {
        int rayIndex = 0;

        for (int v = 0; v < verticalRays; v++)
        {
            float verticalAngle = Mathf.Lerp(minVerticalAngle, maxVerticalAngle, (float)v / (verticalRays - 1));

            for (int h = 0; h < horizontalRays; h++)
            {
                float horizontalAngle = Mathf.Lerp(minAngle, maxAngle, (float)h / (horizontalRays - 1));

                // Calculate ray direction
                Vector3 direction = CalculateRayDirection(horizontalAngle, verticalAngle);

                // Perform raycast
                RaycastHit hit;
                if (Physics.Raycast(transform.position, direction, out hit, maxRange, detectionMask))
                {
                    ranges[rayIndex] = hit.distance;

                    // Visualize ray if enabled
                    if (visualizeRays)
                    {
                        Debug.DrawRay(transform.position, direction * hit.distance, Color.red, rayVisualizationTime);
                    }
                }
                else
                {
                    ranges[rayIndex] = maxRange;

                    // Visualize ray if enabled
                    if (visualizeRays)
                    {
                        Debug.DrawRay(transform.position, direction * maxRange, Color.green, rayVisualizationTime);
                    }
                }

                rayIndex++;
            }
        }

        // Process LiDAR data
        ProcessLiDARData(ranges.ToArray());
    }

    Vector3 CalculateRayDirection(float horizontalAngle, float verticalAngle)
    {
        // Convert angles from degrees to radians
        float hRad = horizontalAngle * Mathf.Deg2Rad;
        float vRad = verticalAngle * Mathf.Deg2Rad;

        // Calculate direction vector
        float x = Mathf.Cos(vRad) * Mathf.Cos(hRad);
        float y = Mathf.Sin(vRad);
        float z = Mathf.Cos(vRad) * Mathf.Sin(hRad);

        // Transform to world space
        return transform.TransformDirection(new Vector3(x, y, z));
    }

    void ProcessLiDARData(float[] ranges)
    {
        // Here you would typically convert to ROS message format
        // or process the data for your specific application
        Debug.Log($"LiDAR: {ranges.Length} points, min: {GetMinRange(ranges)}, max: {GetMaxRange(ranges)}");
    }

    float GetMinRange(float[] ranges)
    {
        float min = float.MaxValue;
        foreach (float range in ranges)
        {
            if (range < min && range > 0)
                min = range;
        }
        return min == float.MaxValue ? 0 : min;
    }

    float GetMaxRange(float[] ranges)
    {
        float max = 0;
        foreach (float range in ranges)
        {
            if (range > max)
                max = range;
        }
        return max;
    }
}
```

### Performance Optimization for LiDAR

For high-resolution LiDAR simulation, optimization is important:

```csharp
using UnityEngine;
using System.Collections.Generic;
using System.Threading.Tasks;

public class OptimizedLiDAR : MonoBehaviour
{
    [Header("Optimization Settings")]
    public int batchSize = 10;  // Number of rays to process in each batch
    public bool useThreading = true;

    private List<Ray> rayBatch;
    private List<RaycastHit> hitResults;
    private bool isProcessing = false;

    void Start()
    {
        rayBatch = new List<Ray>(batchSize);
        hitResults = new List<RaycastHit>(batchSize);
    }

    // Asynchronous LiDAR simulation
    async void SimulateLiDARAsync()
    {
        if (isProcessing) return;
        isProcessing = true;

        // Prepare ray batch
        PrepareRayBatch();

        if (useThreading)
        {
            // Process rays in a separate thread
            var results = await Task.Run(() => ProcessRays(rayBatch));
            ProcessResults(results);
        }
        else
        {
            // Process rays in main thread
            var results = ProcessRays(rayBatch);
            ProcessResults(results);
        }

        isProcessing = false;
    }

    void PrepareRayBatch()
    {
        rayBatch.Clear();

        for (int i = 0; i < batchSize; i++)
        {
            float angle = Mathf.Lerp(-90f, 90f, (float)i / batchSize);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;
            rayBatch.Add(new Ray(transform.position, direction));
        }
    }

    RaycastHit[] ProcessRays(List<Ray> rays)
    {
        RaycastHit[] results = new RaycastHit[rays.Count];

        for (int i = 0; i < rays.Count; i++)
        {
            Physics.Raycast(rays[i], out results[i], 30.0f);
        }

        return results;
    }

    void ProcessResults(RaycastHit[] results)
    {
        // Process the raycast results
        foreach (RaycastHit hit in results)
        {
            if (hit.collider != null)
            {
                Debug.Log($"Hit: {hit.collider.name} at {hit.distance}m");
            }
        }
    }
}
```

## Depth Camera Simulation

Depth cameras provide distance information for each pixel in an image. Here's how to simulate them:

```csharp
using UnityEngine;

public class DepthCameraSimulator : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera depthCamera;
    public RenderTexture depthTexture;
    public int width = 640;
    public int height = 480;
    public float nearClip = 0.1f;
    public float farClip = 10.0f;

    [Header("Output Settings")]
    public bool outputAsPointCloud = false;

    private float[] depthData;

    void Start()
    {
        SetupDepthCamera();
        depthData = new float[width * height];
    }

    void SetupDepthCamera()
    {
        if (depthCamera == null)
            depthCamera = GetComponent<Camera>();

        if (depthCamera != null)
        {
            depthCamera.nearClipPlane = nearClip;
            depthCamera.farClipPlane = farClip;
            depthCamera.depthTextureMode = DepthTextureMode.Depth;

            if (depthTexture == null)
            {
                depthTexture = new RenderTexture(width, height, 24);
                depthTexture.format = RenderTextureFormat.RFloat;
            }

            depthCamera.targetTexture = depthTexture;
        }
    }

    public float[] GetDepthData()
    {
        if (depthCamera == null) return null;

        // Render the scene to get depth data
        depthCamera.Render();

        // Read depth data from render texture
        RenderTexture.active = depthTexture;
        Texture2D depthTex = new Texture2D(width, height, TextureFormat.RFloat, false);
        depthTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        depthTex.Apply();

        Color[] pixels = depthTex.GetPixels();
        for (int i = 0; i < pixels.Length && i < depthData.Length; i++)
        {
            // Depth is stored in the red channel
            depthData[i] = pixels[i].r;
        }

        RenderTexture.active = null;
        DestroyImmediate(depthTex);

        return depthData;
    }

    public Vector3[] ConvertToPointCloud()
    {
        float[] depth = GetDepthData();
        if (depth == null) return null;

        // Calculate field of view parameters
        float fov = depthCamera.fieldOfView * Mathf.Deg2Rad;
        float aspect = (float)width / height;
        float tanHalfFov = Mathf.Tan(fov / 2);

        List<Vector3> points = new List<Vector3>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                float depthValue = depth[index];

                // Only process points within range
                if (depthValue > 0 && depthValue < farClip)
                {
                    // Convert pixel coordinates to normalized device coordinates
                    float normX = (float)x / width * 2 - 1;
                    float normY = (float)y / height * 2 - 1;

                    // Calculate 3D position
                    float z = depthValue;
                    float x3d = normX * z * tanHalfFov * aspect;
                    float y3d = normY * z * tanHalfFov;

                    // Transform from camera space to world space
                    Vector3 point = depthCamera.transform.TransformPoint(new Vector3(x3d, y3d, z));
                    points.Add(point);
                }
            }
        }

        return points.ToArray();
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // This method can be used to apply post-processing effects to depth data
        Graphics.Blit(source, destination);
    }
}
```

## IMU Simulation

An IMU (Inertial Measurement Unit) typically contains accelerometers, gyroscopes, and sometimes magnetometers:

```csharp
using UnityEngine;
using System;

[Serializable]
public class IMUData
{
    public Vector3 linearAcceleration;  // m/s²
    public Vector3 angularVelocity;     // rad/s
    public Vector3 magneticField;       // μT (microtesla)
    public DateTime timestamp;
}

public class IMUSimulator : MonoBehaviour
{
    [Header("Noise Parameters")]
    public float accelerometerNoise = 0.01f;
    public float gyroscopeNoise = 0.001f;
    public float magnetometerNoise = 0.1f;

    [Header("Bias Parameters")]
    public Vector3 accelerometerBias = Vector3.zero;
    public Vector3 gyroscopeBias = Vector3.zero;

    [Header("Sampling Rate")]
    public float updateRate = 100f;  // Hz

    private float nextUpdate = 0f;
    private IMUData currentIMUData;

    void Start()
    {
        currentIMUData = new IMUData();
        currentIMUData.timestamp = DateTime.Now;
    }

    void Update()
    {
        if (Time.time >= nextUpdate)
        {
            SimulateIMU();
            nextUpdate = Time.time + (1f / updateRate);
        }
    }

    void SimulateIMU()
    {
        // Calculate true values based on Unity's physics
        Vector3 trueAcceleration = CalculateTrueAcceleration();
        Vector3 trueAngularVelocity = CalculateTrueAngularVelocity();
        Vector3 trueMagneticField = CalculateTrueMagneticField();

        // Add noise and bias
        currentIMUData.linearAcceleration = AddNoiseAndBias(trueAcceleration, accelerometerNoise, accelerometerBias);
        currentIMUData.angularVelocity = AddNoiseAndBias(trueAngularVelocity, gyroscopeNoise, gyroscopeBias);
        currentIMUData.magneticField = AddNoiseAndBias(trueMagneticField, magnetometerNoise, Vector3.zero);

        currentIMUData.timestamp = DateTime.Now;

        // Process IMU data
        ProcessIMUData(currentIMUData);
    }

    Vector3 CalculateTrueAcceleration()
    {
        // Get acceleration from Unity's physics
        if (GetComponent<Rigidbody>() != null)
        {
            // Calculate acceleration from change in velocity
            // Note: This is a simplified approach
            return GetComponent<Rigidbody>().velocity / Time.fixedDeltaTime;
        }
        else
        {
            // If no rigidbody, use change in transform
            return (transform.position - transform.position) / Time.deltaTime; // This is placeholder
        }
    }

    Vector3 CalculateTrueAngularVelocity()
    {
        // Get angular velocity from Unity's physics
        if (GetComponent<Rigidbody>() != null)
        {
            return GetComponent<Rigidbody>().angularVelocity;
        }
        else
        {
            // Calculate from rotation change
            return transform.rotation.eulerAngles / Time.deltaTime;
        }
    }

    Vector3 CalculateTrueMagneticField()
    {
        // Simulate Earth's magnetic field (approximately 25-65 μT)
        // This is a simplified simulation
        Vector3 magneticNorth = new Vector3(0.2f, 0.0f, 0.5f); // Simplified field
        return transform.InverseTransformDirection(magneticNorth);
    }

    Vector3 AddNoiseAndBias(Vector3 trueValue, float noiseLevel, Vector3 bias)
    {
        Vector3 noise = new Vector3(
            UnityEngine.Random.Range(-noiseLevel, noiseLevel),
            UnityEngine.Random.Range(-noiseLevel, noiseLevel),
            UnityEngine.Random.Range(-noiseLevel, noiseLevel)
        );
        return trueValue + noise + bias;
    }

    void ProcessIMUData(IMUData data)
    {
        // Here you would typically publish to ROS or process for your application
        Debug.Log($"IMU: Accel=({data.linearAcceleration.x:F3}, {data.linearAcceleration.y:F3}, {data.linearAcceleration.z:F3}), " +
                  $"Gyro=({data.angularVelocity.x:F3}, {data.angularVelocity.y:F3}, {data.angularVelocity.z:F3})");
    }

    public IMUData GetLatestIMUData()
    {
        return currentIMUData;
    }
}
```

## Sensor Fusion

Combining multiple sensors for better perception:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SensorFusion : MonoBehaviour
{
    [Header("Sensor References")]
    public SimulatedLiDAR lidar;
    public DepthCameraSimulator depthCamera;
    public IMUSimulator imu;

    [Header("Fusion Parameters")]
    public float fusionRate = 30f;  // Hz
    public float positionUncertainty = 0.1f;
    public float orientationUncertainty = 0.01f;

    private float nextFusionUpdate = 0f;
    private List<IMUData> imuHistory;
    private KalmanFilter poseEstimator;

    void Start()
    {
        imuHistory = new List<IMUData>();
        poseEstimator = new KalmanFilter();
    }

    void Update()
    {
        if (Time.time >= nextFusionUpdate)
        {
            PerformSensorFusion();
            nextFusionUpdate = Time.time + (1f / fusionRate);
        }
    }

    void PerformSensorFusion()
    {
        // Get data from all sensors
        var lidarData = GetLiDARData();
        var depthData = GetDepthData();
        var imuData = imu.GetLatestIMUData();

        // Store IMU history for fusion
        imuHistory.Add(imuData);
        if (imuHistory.Count > 100) // Keep only recent data
            imuHistory.RemoveAt(0);

        // Perform sensor fusion
        var fusedPose = FuseSensors(lidarData, depthData, imuData);

        // Update robot pose estimate
        UpdateRobotPose(fusedPose);
    }

    float[] GetLiDARData()
    {
        if (lidar != null)
        {
            // This would return the actual LiDAR ranges
            return new float[360]; // Placeholder
        }
        return null;
    }

    float[] GetDepthData()
    {
        if (depthCamera != null)
        {
            return depthCamera.GetDepthData();
        }
        return null;
    }

    PoseData FuseSensors(float[] lidarData, float[] depthData, IMUData imuData)
    {
        // Simple sensor fusion algorithm
        // In practice, this would be more sophisticated
        PoseData estimatedPose = new PoseData();

        // Use IMU for orientation (high frequency, drift over time)
        estimatedPose.orientation = IntegrateGyroscope(imuData.angularVelocity);

        // Use LiDAR/depth for position (low frequency, accurate)
        if (lidarData != null)
        {
            estimatedPose.position = EstimatePositionFromLiDAR(lidarData);
        }

        return estimatedPose;
    }

    Vector3 EstimatePositionFromLiDAR(float[] ranges)
    {
        // Simplified position estimation from LiDAR data
        // This would typically involve SLAM algorithms
        return transform.position; // Placeholder
    }

    Quaternion IntegrateGyroscope(Vector3 angularVelocity)
    {
        // Integrate gyroscope data to get orientation
        float dt = 1f / fusionRate;
        Vector3 rotation = angularVelocity * dt;

        // Convert to quaternion
        float angle = rotation.magnitude;
        if (angle > 0)
        {
            Vector3 axis = rotation.normalized;
            return Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
        }

        return Quaternion.identity;
    }

    void UpdateRobotPose(PoseData pose)
    {
        // Update robot's estimated pose based on sensor fusion
        transform.position = pose.position;
        transform.rotation = pose.orientation;
    }
}

[System.Serializable]
public class PoseData
{
    public Vector3 position = Vector3.zero;
    public Quaternion orientation = Quaternion.identity;
}

// Simplified Kalman Filter implementation
public class KalmanFilter
{
    private Matrix4x4 state;      // State vector (position, velocity)
    private Matrix4x4 covariance; // Covariance matrix
    private Matrix4x4 processNoise;
    private Matrix4x4 measurementNoise;

    public KalmanFilter()
    {
        Initialize();
    }

    void Initialize()
    {
        // Initialize state and covariance matrices
        state = Matrix4x4.zero;
        covariance = Matrix4x4.identity;
        processNoise = Matrix4x4.identity * 0.1f;
        measurementNoise = Matrix4x4.identity * 0.5f;
    }

    public void Predict()
    {
        // Prediction step
        // This is a simplified version - full implementation would be more complex
    }

    public void Update(Vector3 measurement)
    {
        // Update step
        // This is a simplified version - full implementation would be more complex
    }
}
```

## Best Practices for Sensor Simulation

1. **Include realistic noise models** that match real sensors
2. **Consider update rates** of different sensors
3. **Implement proper calibration** procedures
4. **Validate against real-world data** when possible
5. **Account for sensor limitations** like range, resolution, and field of view

## Integration with ROS

For ROS integration, you would typically convert sensor data to standard ROS message types:

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using UnityEngine;

public class SensorROSPublisher : MonoBehaviour
{
    private ROSConnection ros;
    private SimulatedLiDAR lidar;
    private IMUSimulator imu;

    private string lidarTopic = "/laser_scan";
    private string imuTopic = "/imu/data";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<LaserScanMsg>(lidarTopic);
        ros.RegisterPublisher<ImuMsg>(imuTopic);

        lidar = GetComponent<SimulatedLiDAR>();
        imu = GetComponent<IMUSimulator>();
    }

    void Update()
    {
        // Publish sensor data periodically
        if (lidar != null)
        {
            var lidarMsg = CreateLaserScanMessage();
            ros.Publish(lidarTopic, lidarMsg);
        }

        if (imu != null)
        {
            var imuMsg = CreateIMUMessage();
            ros.Publish(imuTopic, imuMsg);
        }
    }

    LaserScanMsg CreateLaserScanMessage()
    {
        var msg = new LaserScanMsg();
        msg.header = new std_msgs.Header();
        msg.header.frame_id = "laser_frame";
        msg.header.stamp = new builtin_interfaces.Time();

        // Set laser scan parameters
        msg.angle_min = -Mathf.PI / 2;  // -90 degrees
        msg.angle_max = Mathf.PI / 2;   // 90 degrees
        msg.angle_increment = Mathf.PI / 180; // 1 degree
        msg.time_increment = 0.0f;
        msg.scan_time = 0.1f;
        msg.range_min = 0.1f;
        msg.range_max = 30.0f;

        // Set ranges (this would come from actual LiDAR simulation)
        msg.ranges = new float[181]; // 181 points for -90 to +90 degrees at 1 degree increments
        for (int i = 0; i < msg.ranges.Length; i++)
        {
            msg.ranges[i] = 10.0f; // Placeholder value
        }

        return msg;
    }

    ImuMsg CreateIMUMessage()
    {
        var msg = new ImuMsg();
        msg.header = new std_msgs.Header();
        msg.header.frame_id = "imu_frame";
        msg.header.stamp = new builtin_interfaces.Time();

        // Set orientation (from IMU simulator)
        var imuData = imu.GetLatestIMUData();
        msg.orientation = new geometry_msgs.Quaternion(0, 0, 0, 1); // Placeholder
        msg.angular_velocity = new geometry_msgs.Vector3(
            imuData.angularVelocity.x,
            imuData.angularVelocity.y,
            imuData.angularVelocity.z
        );
        msg.linear_acceleration = new geometry_msgs.Vector3(
            imuData.linearAcceleration.x,
            imuData.linearAcceleration.y,
            imuData.linearAcceleration.z
        );

        return msg;
    }
}
```

## Next Steps

In the next module, we'll explore how AI powers robotic systems with NVIDIA Isaac and related technologies.