import bpy, json

obj = bpy.data.objects.get("Cube")
out = {}
for m in obj.modifiers:
    if m.type != "DISPLACE":
        continue
    minfo = {
        "name": m.name,
        "texture_coords": m.texture_coords,
        "texture_coords_object": m.texture_coords_object.name if m.texture_coords_object else None,
        "direction": m.direction,
        "mid_level": m.mid_level,
        "strength": m.strength,
        "uv_layer": m.uv_layer,
    }
    out.setdefault("modifiers", []).append(minfo)

tex = bpy.data.textures.get("Texture.001")
if tex:
    tinfo = {
        "type": tex.type,
        "has_animation_data": tex.animation_data is not None,
    }
    if tex.animation_data and tex.animation_data.action:
        tinfo["action"] = tex.animation_data.action.name
    for attr in ("noise_scale", "noise_depth", "nabla", "cloud_type", "noise_basis", "noise_type"):
        if hasattr(tex, attr):
            tinfo[attr] = str(getattr(tex, attr))
    out["texture"] = tinfo

empty = bpy.data.objects.get("Empty")
if empty and empty.animation_data and empty.animation_data.action:
    act = empty.animation_data.action
    kf = {}
    def fcurves(act):
        if hasattr(act, "fcurves"):
            return list(act.fcurves)
        fcs = []
        for layer in act.layers:
            for strip in layer.strips:
                for slot in act.slots:
                    cb = strip.channelbag(slot)
                    if cb:
                        fcs.extend(list(cb.fcurves))
        return fcs
    for fc in fcurves(act):
        key = f"{fc.data_path}[{fc.array_index}]"
        kf[key] = [(kp.co[0], kp.co[1]) for kp in fc.keyframe_points]
    out["empty_keyframes"] = kf

print(json.dumps(out, indent=2))
