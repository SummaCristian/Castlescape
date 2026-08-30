// ***** CUSTOM *****

// A corner "fog of war" minimap: a round, top-down view of the dungeon walls,
// always centred on the player and panning with them, that only shows a room
// once the player has walked into it.
//
// It draws nothing itself. Like the crosshair, it's fed to a UiQuad instance:
// every frame main.cpp calls update() with the camera position (which latches
// the room under the player as "visited"), then buildQuads() to get the flat
// rectangle list and hands that to the UiQuad, along with the disc that
// buildQuads() reports back for UiQuad::circleClip (the round mask is applied
// per-pixel in UiQuad.frag). buildQuads() emits, back to front: the backdrop
// disc, a faint floor fill for each visited room, the wall segments belonging
// to those rooms, a dot for any ghost standing in a visited room, and -- dead
// centre -- the player dot with a small nub for their heading.
//
// The wall segments are the XZ footprints of the gameplay colliders, handed
// in once via setWalls() from main.cpp (which already has the merged collider
// list from SceneColliders). A wall shows as soon as any room it touches has
// been visited.
//
// The room boxes and the panel size come from assets/scenes/minimap.json, so
// the map can be re-shaped without touching C++, exactly like colliders.json
// and gameplay.json. "visited" is per-run: restartRun() calls reset().
//
// Same header-only "module" pattern as TextMaker/Scene/UiQuad/CheatHud:
// declarations + implementation in this one file, implementation gated behind
// MINIMAP_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// "modules/Starter.hpp", <json.hpp> and "custom/UiQuad.hpp" (for UiRect) are
// already included by whoever includes this one.

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

// One room the minimap can reveal: an axis-aligned world-space box on the XZ
// plane (Y is ignored, the level is single-storey).
struct MiniMapRoom {
	std::string id;
	std::string name;
	glm::vec2 lo;	// world (minX, minZ)
	glm::vec2 hi;	// world (maxX, maxZ)
};

// One wall segment: the XZ footprint of a collider. Drawn as a thin rectangle.
struct MiniMapWall {
	glm::vec2 lo;	// world (minX, minZ)
	glm::vec2 hi;	// world (maxX, maxZ)
};

struct MiniMap {
	enum Corner { TopLeft, TopRight, BottomLeft, BottomRight };

	// ---- panel layout, overridden from minimap.json "panel" ----
	Corner corner = TopRight;
	float marginPx = 16.0f;
	// The panel is a square; the map is the disc inscribed in it.
	float panelWidthPx = 200.0f;
	float paddingPx = 6.0f;
	// How many world units from the player the edge of the disc reaches. The
	// view is always centred on the player and pans with them.
	float viewRadiusWorld = 20.0f;
	// Soft edge on the disc, in pixels.
	float featherPx = 2.0f;

	// ---- palette (dark panel, faint floor, bright walls) ----
	glm::vec4 backdropColor = {0.04f, 0.04f, 0.06f, 0.62f};
	// Floor fill of a visited room. Alpha low on purpose -- it's just a hint
	// of "you've been here"; set alpha to 0 for a pure wall plan.
	glm::vec4 roomColor      = {1.00f, 1.00f, 1.00f, 0.07f};
	glm::vec4 wallColor      = {0.82f, 0.85f, 0.92f, 0.95f};
	glm::vec4 playerColor    = {1.00f, 0.92f, 0.55f, 1.00f};
	glm::vec4 headingColor   = {1.00f, 0.55f, 0.20f, 1.00f};
	glm::vec4 ghostColor     = {0.60f, 0.28f, 1.00f, 0.95f};

	float playerDotPx = 6.0f;
	float ghostDotPx  = 6.0f;
	// A thin wall must stay at least this many pixels wide/tall or it drops
	// below a pixel and disappears at small panel sizes.
	float wallMinPx = 1.5f;

	// ---- data ----
	std::vector<MiniMapRoom> rooms;
	std::vector<MiniMapWall> walls;
	std::vector<char> visited;			// visited[i] != 0 -> rooms[i] revealed
	glm::vec2 worldLo{0.0f};				// union of every room box, world XZ
	glm::vec2 worldHi{0.0f};

	bool enabled = true;

	// Reads `file`. On failure the minimap is left empty (enabled but with no
	// rooms), so buildQuads() just returns the backdrop.
	void init(const std::string &file);

	// Hands in the wall geometry: the XZ footprint of every collider tall
	// enough to be a wall rather than furniture. Called once from localInit().
	void setWalls(std::vector<MiniMapWall> w) { walls = std::move(w); }

	// Forget every visited room. Called from restartRun().
	void reset();

	// Latch every room whose box contains `camPos` (XZ) as visited. Returns
	// true if this call revealed a room that wasn't visited before.
	bool update(const glm::vec3 &camPos);

	// First room whose box (optionally shrunk by `shrink` world units on every
	// side) contains `p` on the XZ plane, or -1. Used to decide whether a
	// ghost is somewhere the player has already seen.
	int roomAt(const glm::vec3 &p, float shrink = 0.0f) const;

	// The flat-quad list for the UiQuad, as a round view centred on the player:
	// backdrop disc, faint floor of visited rooms, wall segments, ghost dots,
	// then the player dot with a heading nub. Pixel space, top-left origin,
	// matching UiQuad/CheatHud. `outCircle` receives the disc (xy centre px,
	// z radius px, w feather px) to hand to UiQuad::circleClip.
	std::vector<UiRect> buildQuads(int screenW, int screenH,
								   const glm::vec3 &camPos, float camYawDeg,
								   const glm::vec3 *ghostPos, int ghostCount,
								   glm::vec4 &outCircle) const;

	private:
	// Square outer panel, in pixels, for the current screen size.
	void panelRect(int screenW, int screenH,
				   float &ox, float &oy, float &ow, float &oh) const;
};

#ifdef MINIMAP_IMPLEMENTATION

void MiniMap::init(const std::string &file) {
	rooms.clear();
	visited.clear();

	std::ifstream ifs(file);
	if(!ifs.is_open()) {
		std::cout << "MiniMap: '" << file << "' not found, minimap disabled\n";
		return;
	}

	nlohmann::json js;
	try {
		// ignore_comments, like every other scene file here.
		js = nlohmann::json::parse(ifs, nullptr, true, true);
	} catch(const std::exception &e) {
		std::cout << "MiniMap: '" << file << "' failed to parse (" << e.what()
				  << "), minimap disabled\n";
		return;
	}

	if(js.contains("panel")) {
		const auto &p = js["panel"];
		std::string c = p.value("corner", std::string("top-right"));
		if(c == "top-left")           corner = TopLeft;
		else if(c == "top-right")     corner = TopRight;
		else if(c == "bottom-left")   corner = BottomLeft;
		else if(c == "bottom-right")  corner = BottomRight;
		else std::cout << "MiniMap: unknown corner '" << c << "', using top-right\n";

		marginPx         = p.value("marginPx", marginPx);
		panelWidthPx     = p.value("widthPx", panelWidthPx);
		paddingPx        = p.value("paddingPx", paddingPx);
		viewRadiusWorld  = p.value("radiusWorld", viewRadiusWorld);
		featherPx        = p.value("featherPx", featherPx);
	}

	glm::vec2 lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
	for(const auto &r : js.value("rooms", nlohmann::json::array())) {
		const auto &a = r["aabb"];
		if(a.size() != 4) {
			std::cout << "MiniMap: room '" << r.value("id", std::string("?"))
					  << "' has an aabb with " << a.size() << " values, needs 4, skipped\n";
			continue;
		}
		MiniMapRoom m;
		m.id   = r.value("id", std::string(""));
		m.name = r.value("name", std::string(""));
		m.lo = glm::vec2(std::min(a[0].get<float>(), a[2].get<float>()),
						 std::min(a[1].get<float>(), a[3].get<float>()));
		m.hi = glm::vec2(std::max(a[0].get<float>(), a[2].get<float>()),
						 std::max(a[1].get<float>(), a[3].get<float>()));
		rooms.push_back(m);

		lo = glm::min(lo, m.lo);
		hi = glm::max(hi, m.hi);
	}

	visited.assign(rooms.size(), 0);
	if(!rooms.empty()) {
		worldLo = lo;
		worldHi = hi;
	}

	std::cout << "MiniMap: " << rooms.size() << " rooms\n";
}

void MiniMap::reset() {
	std::fill(visited.begin(), visited.end(), (char)0);
}

bool MiniMap::update(const glm::vec3 &camPos) {
	bool revealed = false;
	for(size_t i = 0; i < rooms.size(); i++) {
		if(visited[i]) continue;
		const MiniMapRoom &m = rooms[i];
		if(camPos.x >= m.lo.x && camPos.x <= m.hi.x &&
		   camPos.z >= m.lo.y && camPos.z <= m.hi.y) {
			visited[i] = 1;
			revealed = true;
		}
	}
	return revealed;
}

int MiniMap::roomAt(const glm::vec3 &p, float shrink) const {
	for(size_t i = 0; i < rooms.size(); i++) {
		const MiniMapRoom &m = rooms[i];
		if(p.x >= m.lo.x + shrink && p.x <= m.hi.x - shrink &&
		   p.z >= m.lo.y + shrink && p.z <= m.hi.y - shrink) {
			return (int)i;
		}
	}
	return -1;
}

void MiniMap::panelRect(int screenW, int screenH,
						float &ox, float &oy, float &ow, float &oh) const {
	ow = panelWidthPx;
	oh = panelWidthPx;			// square
	bool right  = (corner == TopRight || corner == BottomRight);
	bool bottom = (corner == BottomLeft || corner == BottomRight);
	ox = right  ? (float)screenW - marginPx - ow : marginPx;
	oy = bottom ? (float)screenH - marginPx - oh : marginPx;
}

std::vector<UiRect> MiniMap::buildQuads(int screenW, int screenH,
									   const glm::vec3 &camPos, float camYawDeg,
									   const glm::vec3 *ghostPos, int ghostCount,
									   glm::vec4 &outCircle) const {
	std::vector<UiRect> out;
	outCircle = glm::vec4(0.0f);
	if(!enabled || rooms.empty()) {
		return out;
	}

	float ox, oy, ow, oh;
	panelRect(screenW, screenH, ox, oy, ow, oh);

	// The disc: centred in the panel, inset by the padding.
	glm::vec2 cpx(ox + ow * 0.5f, oy + oh * 0.5f);
	float radiusPx = std::max(ow * 0.5f - paddingPx, 1.0f);
	outCircle = glm::vec4(cpx.x, cpx.y, radiusPx, featherPx);

	// world XZ -> pixel: fixed scale, always centred on the player. world +X ->
	// +x (right), world +Z -> +y (down), so world -Z (north) is up.
	float pxPerWorld = radiusPx / std::max(viewRadiusWorld, 0.001f);
	glm::vec2 center(camPos.x, camPos.z);
	auto toPx = [&](glm::vec2 xz) {
		return cpx + (xz - center) * pxPerWorld;
	};
	// Cheap reject: is this pixel-space rect fully outside the disc's bounding
	// square? (The shader does the real round clip; this just keeps the mesh
	// small.)
	auto outside = [&](float rx, float ry, float rw, float rh) {
		return rx > cpx.x + radiusPx || rx + rw < cpx.x - radiusPx ||
			   ry > cpx.y + radiusPx || ry + rh < cpx.y - radiusPx;
	};
	auto push = [&](float rx, float ry, float rw, float rh, glm::vec4 col) {
		if(!outside(rx, ry, rw, rh)) out.push_back(UiRect{rx, ry, rw, rh, col});
	};

	// Backdrop: a square covering the disc; the shader rounds it off.
	out.push_back(UiRect{cpx.x - radiusPx, cpx.y - radiusPx,
						 radiusPx * 2.0f, radiusPx * 2.0f, backdropColor});

	bool anyVisited = std::any_of(visited.begin(), visited.end(),
								  [](char v){ return v != 0; });
	if(!anyVisited) {
		return out;
	}

	// Faint floor fill for each visited room.
	for(size_t i = 0; i < rooms.size(); i++) {
		if(!visited[i]) continue;
		glm::vec2 a = toPx(rooms[i].lo);
		glm::vec2 b = toPx(rooms[i].hi);
		push(a.x, a.y, b.x - a.x, b.y - a.y, roomColor);
	}

	// Wall segments: a wall shows once any room whose box it touches has been
	// visited.
	for(const MiniMapWall &w : walls) {
		bool show = false;
		for(size_t i = 0; i < rooms.size() && !show; i++) {
			if(!visited[i]) continue;
			const MiniMapRoom &r = rooms[i];
			show = (w.lo.x <= r.hi.x && w.hi.x >= r.lo.x &&
					w.lo.y <= r.hi.y && w.hi.y >= r.lo.y);
		}
		if(!show) continue;

		glm::vec2 a = toPx(w.lo);
		glm::vec2 b = toPx(w.hi);
		float rx = a.x, ry = a.y, rw = b.x - a.x, rh = b.y - a.y;
		if(rw < wallMinPx) { rx -= (wallMinPx - rw) * 0.5f; rw = wallMinPx; }
		if(rh < wallMinPx) { ry -= (wallMinPx - rh) * 0.5f; rh = wallMinPx; }
		push(rx, ry, rw, rh, wallColor);
	}

	// Ghost dots -- only while the ghost is in a room the player has seen.
	for(int g = 0; g < ghostCount; g++) {
		int rm = roomAt(ghostPos[g]);
		if(rm < 0 || !visited[rm]) continue;
		glm::vec2 p = toPx(glm::vec2(ghostPos[g].x, ghostPos[g].z));
		push(p.x - ghostDotPx * 0.5f, p.y - ghostDotPx * 0.5f,
			 ghostDotPx, ghostDotPx, ghostColor);
	}

	// Player is always dead centre. Heading nub first, dot on top.
	float yaw = glm::radians(camYawDeg);
	// front on the XZ plane is (cos yaw, sin yaw) -- see GameLogic()'s front.
	glm::vec2 dir(std::cos(yaw), std::sin(yaw));
	float nub = playerDotPx * 0.9f;
	glm::vec2 nc = cpx + dir * (playerDotPx * 0.7f);
	out.push_back(UiRect{nc.x - nub * 0.5f, nc.y - nub * 0.5f,
						 nub, nub, headingColor});
	out.push_back(UiRect{cpx.x - playerDotPx * 0.5f, cpx.y - playerDotPx * 0.5f,
						 playerDotPx, playerDotPx, playerColor});

	return out;
}

#endif
