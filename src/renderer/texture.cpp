#include "texture.h"

#include "stb_image/stb_image.h"
#include "glerrorhandler.h"

Texture::Texture(const std::string& path)
	: m_RendererID(0), m_FilePath(path), m_LocalBuffer(nullptr), m_Width(0), m_Height(0), m_BPP(0)
{
	stbi_set_flip_vertically_on_load(true);  // image coords usually set 0,0 at top left, but OpenGL likes bottom left, so we flip vertically on load
	m_LocalBuffer = stbi_load(path.c_str(), &m_Width, &m_Height, &m_BPP, 4);

	GLCall(glGenTextures(1, &m_RendererID));
	GLCall(glBindTexture(GL_TEXTURE_2D, m_RendererID));

	/* Setting up texture parameters */

	// Setting texture filtering for minification (many texels to single pixel) and magnification (single texel to many pixels)
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	// setting up texture wrap for 2d textures
	// GL_CLAMP_TO_EDGE -> when the texture is used but not scaled enough to cover all, it paints with the pixels of the edges the rest of the polygon
	// GL_REPEAT -> when the same case, it repeats the texture over and over again filling the rest of the polygon
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));  // s,t is like x,y, but in textures
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	/* Set the texture */
	// internal format: how opengl stores the data
	// format: the format of the data we're providing
	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_LocalBuffer));

	// Unbind
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));

	// In our case, we want to release the memory of the image, since it's on gpu now, but we may want to store it for other purposes
	if (m_LocalBuffer)
		stbi_image_free(m_LocalBuffer);

}

Texture::~Texture()
{
	stbi_image_free(m_LocalBuffer);
	GLCall(glDeleteTextures(1, &m_RendererID));
}

void Texture::Bind(unsigned int slot) const
{
	GLCall(glActiveTexture(GL_TEXTURE0 + slot));  // Set GL_TEXTUREslot as active. OpenGL allows 32, but graphics cards may handle only 8. Apparently, is not usual to actually use many, even 8.
	GLCall(glBindTexture(GL_TEXTURE_2D, m_RendererID));
}

void Texture::Unbind() const
{
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}