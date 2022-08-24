#version 450

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 iColor;

layout (location = 0) out vec4 oColor;

void main()
{
	oColor = vec4(iColor, 1.0);
}
