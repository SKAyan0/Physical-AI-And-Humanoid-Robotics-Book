---
sidebar_position: 3
---

# Chapter 2.2: Visual Fidelity: Rendering and Human-Robot Interaction in Unity

## Introduction to Unity for Robotics

Unity is a powerful game engine that has found significant applications in robotics simulation. Its high-quality rendering capabilities and physics engine make it ideal for creating photorealistic environments for robot training and testing.

## Unity Robotics Setup

### Installing Unity Robotics Tools

To work with robotics in Unity, you'll need to install the Unity Robotics Hub, which includes:

- Unity Robotics Package
- Unity Simulation Package
- ML-Agents Toolkit
- Perception Package

### Basic Robot Integration

Here's how to set up a basic robot in Unity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityRobotController : MonoBehaviour
{
    // ROS Connector
    private ROSConnection ros;

    // Robot joints
    public HingeJoint[] joints;

    // Topic names
    private string jointCommandTopic = "/joint_commands";
    private string sensorTopic = "/sensor_data";

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(jointCommandTopic);

        // Subscribe to sensor data
        ros.Subscribe<JointStateMsg>(sensorTopic, OnSensorDataReceived);
    }

    void OnSensorDataReceived(JointStateMsg msg)
    {
        // Process sensor data
        for (int i = 0; i < joints.Length && i < msg.position.Length; i++)
        {
            // Update joint positions based on sensor data
            joints[i].GetComponent<Rigidbody>().MovePosition(
                CalculateNewPosition(joints[i], msg.position[i])
            );
        }
    }

    void Update()
    {
        // Send joint commands to ROS
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SendJointCommands();
        }
    }

    void SendJointCommands()
    {
        // Create and send joint state message
        var jointState = new JointStateMsg();
        jointState.name = new string[] { "joint1", "joint2", "joint3" };
        jointState.position = new double[] { 0.5, 1.0, -0.5 };

        ros.Publish(jointCommandTopic, jointState);
    }

    Vector3 CalculateNewPosition(HingeJoint joint, double targetPosition)
    {
        // Calculate new position based on joint target
        return joint.transform.position +
               joint.axis * (float)targetPosition * 0.1f;
    }
}
```

## High-Fidelity Rendering

### Materials and Shaders

Creating realistic robot models requires attention to materials and lighting:

```csharp
using UnityEngine;

public class RobotMaterialController : MonoBehaviour
{
    public Material[] robotMaterials;
    public Light environmentLight;

    [Header("Material Properties")]
    public Color baseColor = Color.gray;
    public float metallic = 0.8f;
    public float smoothness = 0.6f;

    void Start()
    {
        ApplyMaterialProperties();
    }

    void ApplyMaterialProperties()
    {
        foreach (Material mat in robotMaterials)
        {
            mat.SetColor("_Color", baseColor);
            mat.SetFloat("_Metallic", metallic);
            mat.SetFloat("_Smoothness", smoothness);
        }
    }

    public void UpdateMaterialProperties(Color newColor, float newMetallic, float newSmoothness)
    {
        baseColor = newColor;
        metallic = newMetallic;
        smoothness = newSmoothness;
        ApplyMaterialProperties();
    }
}
```

### Lighting Setup

For photorealistic rendering, proper lighting is crucial:

```csharp
using UnityEngine;

public class LightingSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light mainLight;
    public Light[] fillLights;
    public ReflectionProbe reflectionProbe;
    public Skybox skyboxMaterial;

    [Header("Environment Settings")]
    public float exposure = 1.0f;
    public float ambientIntensity = 0.2f;
    public Color ambientColor = Color.white;

    void Start()
    {
        ConfigureLighting();
    }

    void ConfigureLighting()
    {
        // Configure main directional light
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.intensity = 1.0f;
            mainLight.shadows = LightShadows.Soft;
        }

        // Configure fill lights
        foreach (Light fillLight in fillLights)
        {
            fillLight.intensity = 0.3f;
            fillLight.color = Color.gray;
        }

        // Set ambient lighting
        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientLight = ambientColor;

        // Update reflection probe
        if (reflectionProbe != null)
        {
            reflectionProbe.RenderProbe();
        }
    }
}
```

## Human-Robot Interaction

### Interaction System

Creating intuitive human-robot interaction in Unity:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class HumanRobotInteraction : MonoBehaviour
{
    [Header("UI Elements")]
    public Canvas interactionCanvas;
    public Button[] robotButtons;
    public Slider[] robotSliders;
    public Text statusText;

    [Header("Interaction Settings")]
    public float interactionDistance = 3.0f;
    public LayerMask robotLayer;

    private GameObject currentRobot;
    private bool isInteracting = false;

    void Update()
    {
        HandleInteraction();
    }

    void HandleInteraction()
    {
        if (Input.GetKeyDown(KeyCode.E))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, interactionDistance, robotLayer))
            {
                if (hit.collider.CompareTag("Robot"))
                {
                    StartInteraction(hit.collider.gameObject);
                }
            }
        }
    }

    void StartInteraction(GameObject robot)
    {
        currentRobot = robot;
        isInteracting = true;

        // Enable interaction UI
        interactionCanvas.gameObject.SetActive(true);
        UpdateStatus("Interacting with robot: " + robot.name);

        // Set up UI callbacks
        SetupUICallbacks();
    }

    void SetupUICallbacks()
    {
        // Button interactions
        foreach (Button button in robotButtons)
        {
            button.onClick.AddListener(() => OnRobotButtonClicked(button.name));
        }

        // Slider interactions
        foreach (Slider slider in robotSliders)
        {
            slider.onValueChanged.AddListener((value) => OnRobotSliderChanged(slider.name, value));
        }
    }

    void OnRobotButtonClicked(string buttonName)
    {
        switch (buttonName)
        {
            case "MoveForwardButton":
                currentRobot.SendMessage("MoveForward");
                break;
            case "StopButton":
                currentRobot.SendMessage("StopMovement");
                break;
            case "GripperOpenButton":
                currentRobot.SendMessage("OpenGripper");
                break;
            case "GripperCloseButton":
                currentRobot.SendMessage("CloseGripper");
                break;
        }
    }

    void OnRobotSliderChanged(string sliderName, float value)
    {
        // Send command to robot based on slider value
        if (currentRobot != null)
        {
            currentRobot.SendMessage("SetJointPosition", new JointPositionData {
                jointName = sliderName,
                position = value
            });
        }
    }

    void UpdateStatus(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
    }

    public void EndInteraction()
    {
        isInteracting = false;
        interactionCanvas.gameObject.SetActive(false);
        currentRobot = null;
        UpdateStatus("Interaction ended");
    }
}

[System.Serializable]
public class JointPositionData
{
    public string jointName;
    public float position;
}
```

## Perception and Sensor Simulation

### Camera System

Simulating RGB and depth cameras in Unity:

```csharp
using UnityEngine;

public class RobotCameraSystem : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera rgbCamera;
    public Camera depthCamera;
    public RenderTexture rgbTexture;
    public RenderTexture depthTexture;

    [Header("Sensor Parameters")]
    public float fieldOfView = 60.0f;
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;
    public float nearClip = 0.1f;
    public float farClip = 10.0f;

    void Start()
    {
        ConfigureCameras();
    }

    void ConfigureCameras()
    {
        // Configure RGB camera
        if (rgbCamera != null)
        {
            rgbCamera.fieldOfView = fieldOfView;
            rgbCamera.nearClipPlane = nearClip;
            rgbCamera.farClipPlane = farClip;
            rgbCamera.targetTexture = rgbTexture;
        }

        // Configure depth camera
        if (depthCamera != null)
        {
            depthCamera.fieldOfView = fieldOfView;
            depthCamera.nearClipPlane = nearClip;
            depthCamera.farClipPlane = farClip;
            depthCamera.targetTexture = depthTexture;

            // Set depth camera to render depth
            depthCamera.depthTextureMode = DepthTextureMode.Depth;
        }
    }

    public Texture2D GetRGBImage()
    {
        if (rgbCamera != null && rgbTexture != null)
        {
            RenderTexture.active = rgbTexture;
            Texture2D image = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RGB24, false);
            image.ReadPixels(new Rect(0, 0, resolutionWidth, resolutionHeight), 0, 0);
            image.Apply();
            RenderTexture.active = null;
            return image;
        }
        return null;
    }

    public float[] GetDepthData()
    {
        if (depthCamera != null && depthTexture != null)
        {
            RenderTexture.active = depthTexture;
            Texture2D depthImage = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RFloat, false);
            depthImage.ReadPixels(new Rect(0, 0, resolutionWidth, resolutionHeight), 0, 0);
            depthImage.Apply();

            Color[] pixels = depthImage.GetPixels();
            float[] depthValues = new float[pixels.Length];

            for (int i = 0; i < pixels.Length; i++)
            {
                depthValues[i] = pixels[i].r; // Depth is stored in the red channel
            }

            RenderTexture.active = null;
            Destroy(depthImage);
            return depthValues;
        }
        return null;
    }
}
```

## Best Practices

1. **Use appropriate physics materials** for realistic interactions
2. **Optimize rendering performance** with Level of Detail (LOD) systems
3. **Implement proper collision detection** for safe robot operation
4. **Use occlusion culling** to improve rendering performance
5. **Configure lighting settings** to match real-world conditions

## Integration with ROS

Unity can communicate with ROS systems using the ROS-TCP-Connector:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityROSBridge : MonoBehaviour
{
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>("/unity_status");
        ros.Subscribe<StringMsg>("/unity_commands", OnUnityCommand);
    }

    void OnUnityCommand(StringMsg cmd)
    {
        Debug.Log("Received command: " + cmd.data);

        // Process the command
        ProcessCommand(cmd.data);
    }

    void ProcessCommand(string command)
    {
        switch (command)
        {
            case "start_simulation":
                StartSimulation();
                break;
            case "stop_simulation":
                StopSimulation();
                break;
            case "reset_environment":
                ResetEnvironment();
                break;
        }

        // Send confirmation back to ROS
        var status = new StringMsg();
        status.data = "Command processed: " + command;
        ros.Publish("/unity_status", status);
    }

    void StartSimulation()
    {
        Debug.Log("Starting Unity simulation");
        // Implementation for starting simulation
    }

    void StopSimulation()
    {
        Debug.Log("Stopping Unity simulation");
        // Implementation for stopping simulation
    }

    void ResetEnvironment()
    {
        Debug.Log("Resetting Unity environment");
        // Implementation for resetting environment
    }
}
```

## Next Steps

In the next chapter, we'll explore simulating various sensor types like LiDAR, depth cameras, and IMUs in Unity environments.