/*******************************************************************************************************************
* FILE NAME   :    tools_logo.h
*
* PROJ NAME   :    general use
*
* DESCRIPTION :    a tool creates a logo data
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2022 MAY 12      Yu Liu          Creation
*
********************************************************************************************************************/
#pragma once
#include "all_common.h"

namespace tools_logo
{
#define STBI_WINDOWS_UTF8

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_DEFINE
#include "stb/stb.h"
    
    inline bool creator(const char* fn)
    {
        int w, h, n;
        int res = stbi_info(fn, &w, &h, &n);
        uint8* data = stbi_load(fn, &w, &h, &n, 4);
        int size = w * h * n;
        FILE* fp;
        if (fopen_s(&fp, "logo_data.h", "wb")) {
            printf("file doesn't exist: %s\n", fn);
            return false;
        }
        fprintf(fp, "#pragma once\n\n");
        fprintf(fp, "namespace logo_image\n{\n    ");
        fprintf(fp, "static int width = %d, height = %d, channel = %d;\n    ", w, h, n);
        fprintf(fp, "static const unsigned char data[] = {\n        ");
        for (int i = 0, k = 0; k < size; k++) {
            if (k < size - 1) {
                fprintf(fp, "%d, ", data[k]);
                i++;
                if (i == w)
                    i = 0, fprintf(fp, "\n        ");
            }
            else
                fprintf(fp, "%d\n    };\n}\n", data[k]);
        }
        fclose(fp);
    }

    inline bool creator(const char* image_fn, const char* data_fn)
    {
        int w, h, n;
        int res = stbi_info(image_fn, &w, &h, &n);
        uint8* data = stbi_load(image_fn, &w, &h, &n, 4);
        int size = w * h * n;
        FILE* fp;
        char tmp[200];
        sprintf_s(tmp, "%s.h", data_fn);
        if (fopen_s(&fp, tmp, "wb")) {
            printf("file doesn't exist: %s\n", image_fn);
            return false;
        }
        fprintf(fp, "#pragma once\n\n");
        fprintf(fp, "namespace %s\n{\n    ", data_fn);
        fprintf(fp, "static int width = %d, height = %d, channel = %d;\n    ", w, h, n);
        fprintf(fp, "static const unsigned char data[] = {\n        ");
        for (int i = 0, k = 0; k < size; k++) {
            if (k < size - 1) {
                fprintf(fp, "%d, ", data[k]);
                i++;
                if (i == w)
                    i = 0, fprintf(fp, "\n        ");
            }
            else
                fprintf(fp, "%d\n    };\n}\n", data[k]);
        }
        fclose(fp);
    }
}
