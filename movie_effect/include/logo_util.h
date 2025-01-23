/*******************************************************************************************************************
* FILE NAME   :    logo_util.h
*
* PROJ NAME   :    general use
*
* DESCRIPTION :    a class creates a logo
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2022 MAY 12      Yu Liu          Creation
*
********************************************************************************************************************/
#pragma once
#include "ckey_common.h"
#include "logo_data.h"

class LogoTex
{
    float m_texX[2], m_texY[2];
    int m_imgHSize, m_imgVSize, m_imgNChnl;
    GLuint m_texHandle = 0;
    GLuint m_vaoHandle = 0;
    GLuint m_vboHandle = 0;
    std::vector<float>  m_vtxArray;
    std::vector<GLuint> m_elmArray;

    void buffer(const unsigned char* data)
    {
        if (!m_vaoHandle)
            glGenVertexArrays(1, &m_vaoHandle);
        if (!m_vboHandle)
            glGenBuffers(1, &m_vboHandle);

        glBindVertexArray(m_vaoHandle);

        glBindBuffer(GL_ARRAY_BUFFER, m_vboHandle);
        glBufferData(GL_ARRAY_BUFFER, m_vtxArray.size() * sizeof(float), m_vtxArray.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        if (!m_texHandle)
            glGenTextures(1, &m_texHandle);
        glBindTexture(GL_TEXTURE_2D, m_texHandle);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_imgHSize, m_imgVSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glBindVertexArray(0);
    }

public:
    LogoTex(const float x0, const float y0, const float xs, const float ys)
    {
        m_texX[0] = x0, m_texX[1] = x0 + xs;
        m_texY[0] = y0, m_texY[1] = y0 + ys;

        m_imgHSize = logo_image::width;
        m_imgVSize = logo_image::height;
        m_imgNChnl = logo_image::channel;

        m_vtxArray.assign({
            // positions: x,y,z         normals: x,y,z   texture coords
            m_texX[1], m_texY[1], 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, // top right
            m_texX[1], m_texY[0], 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // bottom right
            m_texX[0], m_texY[0], 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // bottom left
            m_texX[0], m_texY[0], 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // bottom left
            m_texX[0], m_texY[1], 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // top left 
            m_texX[1], m_texY[1], 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f  // top right
            });

        this->buffer(logo_image::data);
    }

    ~LogoTex()
    {
        glDeleteVertexArrays(1, &m_texHandle);
        glDeleteVertexArrays(1, &m_vaoHandle);
        glDeleteBuffers(1, &m_vboHandle);
    }

    void update()
    {
        buffer(logo_image::data);
    }

    void render(void)
    {
        if (!m_vaoHandle)
            return;
        glBindVertexArray(m_vaoHandle);
        glBindTexture(GL_TEXTURE_2D, m_texHandle);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }
};