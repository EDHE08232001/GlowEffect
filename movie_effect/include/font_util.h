/*******************************************************************************************************************
* FILE NAME   :    font_util.h
*
* PROJ NAME   :    general use
*
* DESCRIPTION :    a class creates fonts
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2022 MAY 18      Yu Liu          Creation
*
********************************************************************************************************************/
#pragma once
#include "ckey_common.h"
#include "glsl_prog.h"
#include "ft2build.h"
#include FT_FREETYPE_H


class FontTex
{
    struct FontChar {
        unsigned int TexID;  // ID handle of the glyph texture
        glm::ivec2   Size;       // Size of glyph
        glm::ivec2   Bearing;    // Offset from baseline to left/top of glyph
        unsigned int Advance;    // Offset to advance to next glyph
    };

    int m_hSize, m_vSize, m_FontHeight;
    vec3f_t m_bgColor;
    GLFWwindow* m_glfwWindow = nullptr;
    glslProgram* m_glslProgram = nullptr;
    std::map<char, FontChar> m_CharTable;
    unsigned int m_vaoHandle = 0, m_vboHandle = 0;

    std::string  font_vert = R"(
    #version 400 core
    layout(location = 0) in vec4 vertex;
    out vec2 TexCoords;
    uniform mat4 projection;
    void main()
    {
        gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
        TexCoords = vertex.zw;
    }
    )";

    std::string font_frag = R"(
    #version 400 core
    in vec2 TexCoords;
    out vec4 color;

    uniform sampler2D text;
    uniform vec3 textColor;

    void main()
    {
        vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
        color = vec4(textColor, 1.0) * sampled;
    }
    )";


    void load_font(const char* font_nm)
    {
        FT_Library ft_lib;
        // All functions return a value different than 0 whenever an error occurred
        if (FT_Init_FreeType(&ft_lib)) {
            printf("ERROR::FREETYPE: Could not init FreeType Library");
        }

        // load font as face
        FT_Face face;
        if (FT_New_Face(ft_lib, font_nm, 0, &face)) {
            printf("ERROR::FREETYPE: Failed to load font %s\n", font_nm);
            exit(0);
        }
        else {
            // set size to load glyphs as
            FT_Set_Pixel_Sizes(face, 0, m_FontHeight);

            // disable byte-alignment restriction
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            // load first 128 characters of ASCII set
            for (unsigned char c = 0; c < 128; c++)
            {
                // Load character glyph 
                if (FT_Load_Char(face, c, FT_LOAD_RENDER))
                {
                    printf("ERROR::FREETYTPE: failed to load glyph %d\n", c);
                    continue;
                }
                // generate texture
                unsigned int texture;
                glGenTextures(1, &texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D,
                    0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows,
                    0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer
                );
                // set texture options
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // GL_LINEAR_MIPMAP_LINEAR);// 
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // GL_LINEAR_MIPMAP_LINEAR);// 
                //glGenerateMipmap(GL_TEXTURE_2D); // no difference to linear

                // now store character for later use
                FontChar character = {
                    texture,
                    glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                    glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                    static_cast<unsigned int>(face->glyph->advance.x)
                };
                m_CharTable.insert(std::pair<char, FontChar>(c, character));
            }
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        // destroy FreeType once we're finished
        FT_Done_Face(face);
        FT_Done_FreeType(ft_lib);
    }

    void gen_buffer(void)
    {
        // set background color
        glClearColor(m_bgColor.r, m_bgColor.g, m_bgColor.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glGenVertexArrays(1, &m_vaoHandle);
        glGenBuffers(1, &m_vboHandle);
        glBindVertexArray(m_vaoHandle);
        glBindBuffer(GL_ARRAY_BUFFER, m_vboHandle);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }


public:
    FontTex(GLFWwindow* win, const int w, const int h, const char* font_nm, const uint32_t font_sz, const vec3f_t bg_color) :
        m_glfwWindow(win), m_hSize(w), m_vSize(h), m_FontHeight(font_sz < 6 ? 6 : font_sz), m_bgColor(bg_color)
    {
        m_glslProgram = new glslProgram();
        m_glslProgram->compile(font_vert, glsl::VERTEX, "font.vert");
        m_glslProgram->compile(font_frag, glsl::FRAGMENT, "font.frag");
        m_glslProgram->link();
        m_glslProgram->use();

        this->load_font(font_nm);
        this->gen_buffer();
    }

    ~FontTex()
    {
        for (auto c : m_CharTable) {
            if (c.second.TexID)
                glDeleteVertexArrays(1, &c.second.TexID);
        }

        if (m_vaoHandle)
            glDeleteVertexArrays(1, &m_vaoHandle);
        
        if (m_vboHandle)
            glDeleteBuffers(1, &m_vboHandle);
        
        if (m_glslProgram)
            delete m_glslProgram;
    }

    void render(const std::string text, const float x0, const float y0, const float scale, const glm::vec3 color)
    {
        // activate current window
        glfwMakeContextCurrent(m_glfwWindow);
        //glfwGetFramebufferSize(m_glfwWindow, &m_hSize, &m_vSize);
        //glViewport(0, 0, m_hSize, m_vSize);

        m_glslProgram->use();
        m_glslProgram->set_uniform("textColor", color);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(m_vaoHandle);
        glClearColor(m_bgColor.r, m_bgColor.g, m_bgColor.b, 1.0f);  // define background color 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);         // set defined color

        glm::mat4 project = glm::ortho(0.0f, static_cast<float>(m_hSize), 0.0f, static_cast<float>(m_vSize));
        if (m_glslProgram->set_uniform("projection", project))
            printf("Err: project failed\n");

        // iterate through all characters
        float x = x0, y = y0;
        std::string::const_iterator c;
        for (c = text.begin(); c != text.end(); c++)
        {
            if (*c == 10) {
                y += m_FontHeight * scale + 1.f;    // margin = 1.f
                x = x0;
                continue;
            }

            FontChar font_char = m_CharTable[*c];

            float xpos = x + font_char.Bearing.x * scale;
            float ypos = m_vSize - y - (font_char.Size.y - font_char.Bearing.y) * scale;

            float w = font_char.Size.x * scale;
            float h = font_char.Size.y * scale;
            // update VBO for each character
            if (w != 0.f && h != 0.f) {
                float vertices[6][4] = {
                    { xpos,     ypos + h,   0.0f, 0.0f },
                    { xpos,     ypos,       0.0f, 1.0f },
                    { xpos + w, ypos,       1.0f, 1.0f },

                    { xpos,     ypos + h,   0.0f, 0.0f },
                    { xpos + w, ypos,       1.0f, 1.0f },
                    { xpos + w, ypos + h,   1.0f, 0.0f }
                };
                // render glyph texture over quad
                glBindTexture(GL_TEXTURE_2D, font_char.TexID);
                // update content of VBO memory
                glBindBuffer(GL_ARRAY_BUFFER, m_vboHandle);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                // render quad
                glDrawArrays(GL_TRIANGLES, 0, 6);
            }
            // now advance cursors for next glyph (note that advance is number of 1/64 pixels)
            x += (font_char.Advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64)
        }
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glfwSwapBuffers(m_glfwWindow);
    }

};