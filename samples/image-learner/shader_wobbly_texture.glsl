#version 330 core

in vec2 UV;

out vec3 color;

uniform sampler2D rendered_texture;
uniform float time;

void main() {

	// color = texture(rendered_texture, UV + 0.005 * vec2(sin(time + 1024.0 * UV.x), cos(time + 768.0 * UV.y))).xyz;
	color = texture(rendered_texture, UV).xyz;
	// color = vec3(0.5, 0.5, 0.5);

}
