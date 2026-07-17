using UnityEngine;

public class RigProbe : MonoBehaviour
{
    private Animator humanoidRig;
    private Transform armBone;

    void Start()
    {
        humanoidRig = GetComponent<Animator>();
        armBone = humanoidRig.GetBoneTransform(HumanBodyBones.RightUpperArm);
        if (armBone == null)
        {
            enabled = false;
            Debug.LogError("RightUpperArm is not found");
            return;
        }

        Debug.Log("Bone Name = " + armBone.name);
        Debug.Log("localRotation = " + armBone.localRotation);
    }

    void Update()
    {
        armBone.localRotation = Quaternion.Euler(0, 0, Mathf.Sin(Time.time) * 45f);
        Debug.Log("localRotation = " + armBone.localRotation);
    }
}
