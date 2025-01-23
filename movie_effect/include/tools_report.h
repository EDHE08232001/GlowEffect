/*******************************************************************************************************************
 * FILE NAME   :    tools_report.h
 *
 * PROJECTION  :    2.1D DVE
 *
 * DESCRIPTION :    a file reports various singular cases
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2017 DEC 06      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#ifndef __TOOLS_REPORT_H__
#define __TOOLS_REPORT_H__

#include "all_common.h"
using namespace std;

namespace rpt{
    inline void ERROR_report( const char *name ){

        string  input;
        std::cout << "ERROR: " << name << std::endl;
        std::cout << "Stop? ";
        std::getline( cin, input );
        exit( 0 );
    }

    inline void ERROR_report( const char *name, const int line_num, const int pix_cnt ){

        string  input;
        std::cout << dec;
        std::cout << "ERROR: " << name << " at line " << line_num << ", pixel " << pix_cnt << std::endl;
        std::cout << "Stop? ";
        std::getline( cin, input );
        exit( 0 );
    }

    inline void ERROR_report( const char *name, const int line_num, const int pix_cnt, const int err_int ){

        string  input;
        std::cout << dec;
        std::cout << "ERROR: " << name << " at line " << line_num << ", pixel " << pix_cnt << ", error " << err_int << std::endl;
        std::cout << "Stop? ";
        std::getline( cin, input );
        exit( 0 );
    }

    inline void ERROR_report( const char *name, const int line_num, const int pix_cnt, const float err_flt ){

        string  input;
        std::cout << dec;
        std::cout << "ERROR: " << name << " at line " << line_num << ", pixel " << pix_cnt << ", error " << err_flt << std::endl;
        std::cout << "Stop? ";
        std::getline( cin, input );
        exit( 0 );
    }

    inline void ERROR_nonstop( const char *name ){

        string  input;
        std::cout << "ERROR: " << name << std::endl;
        std::cout << "Wait? ";
        std::getline( cin, input );
    }

    inline void ERROR_nonstop( const char *name, const int line_num, const int pix_cnt ){

        string  input;
        std::cout << dec;
        std::cout << "ERROR: " << name << " at line " << line_num << ", pixel " << pix_cnt << std::endl;
        std::cout << "Wait? ";
        std::getline( cin, input );
    }

    inline void WARNING_report( const char *name ){

        string  input;
        std::cout << "WARNING: " << name << std::endl;
    }

    inline void WARNING_report( const char *name, const int line_num, const int pix_cnt, const int int_diff, const float flt_diff ){

        string  input;
        std::cout << dec;
        std::cout << "WARNING: " << name << " at line " << line_num << ", pixel " << pix_cnt << ", diff ";
        if( int_diff )
            std::cout << int_diff << std::endl;
        else if( flt_diff )
            std::cout << flt_diff << std::endl;
        else
            std::cout << std::endl;
    }

}

#endif
