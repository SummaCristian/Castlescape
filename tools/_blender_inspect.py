import bpy, json

def get_fcurves(act):
    # Blender >= 4.4 "layered" action API; older versions have act.fcurves directly.
    if hasattr(act, "fcurves"):
        try:
            return list(act.fcurves)
        except Exception:
            pass
    fcs = []
    try:
        for layer in act.layers:
            for strip in layer.strips:
                for slot in act.slots:
                    try:
                        cb = strip.channelbag(slot)
                    except Exception:
                        cb = None
                    if cb:
                        fcs.extend(list(cb.fcurves))
    except Exception as e:
        fcs = [("ERROR", str(e))]
    return fcs

out = {}
out["objects"] = []
for obj in bpy.data.objects:
    info = {
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "animation_data": None,
    }
    if obj.animation_data and obj.animation_data.action:
        act = obj.animation_data.action
        fcs = get_fcurves(act)
        fc_info = []
        for fc in fcs:
            if isinstance(fc, tuple):
                fc_info.append({"error": fc[1]})
            else:
                fc_info.append({"data_path": fc.data_path, "array_index": fc.array_index, "n_keyframes": len(fc.keyframe_points)})
        info["animation_data"] = {
            "action": act.name,
            "frame_range": list(act.frame_range),
            "fcurves": fc_info,
        }
    out["objects"].append(info)

out["actions"] = [a.name for a in bpy.data.actions]
out["scene_fps"] = bpy.context.scene.render.fps
out["frame_start"] = bpy.context.scene.frame_start
out["frame_end"] = bpy.context.scene.frame_end
out["blender_version"] = bpy.app.version_string

print(json.dumps(out, indent=2))
