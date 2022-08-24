#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in vec3 iColor;

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projcam;
} uScene;

layout (location = 0) out vec2 v2fTexCoord;
layout (location = 1) out vec3 oColor;

void main()
{
	v2fTexCoord = texCoord;
	oColor = iColor;
	gl_Position = uScene.projcam * vec4(position, 1.0f);
}
