/*******************************************************************************************************************
* FILE NAME   :    alg_laguerre.h
*
* PROJECTION  :    general algebraic tools
*
* DESCRIPTION :    a file implements laguerre algorithm to find roost of polynomial
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2020 OCT 15      Yu Liu          Creation
*
********************************************************************************************************************/
#pragma once
#include "all_common.h"

namespace alg
{
    class laguerre_t
    {
        typedef     std::complex<double>    complex_t;

    public:
        laguerre_t(const int degree)
        {
            N = degree;
            m_coef = new complex_t[degree + 1];
            m_root = new complex_t[degree + 1];
        }

        ~laguerre_t()
        {
            if (m_coef)  delete[] m_coef;
            if (m_root)  delete[] m_root;
        }

    private:
        void solver(const bool polish = true, const bool do_sort = true)
        {

        }

        void sorting(void)
        {

        }

        void evaluate(void)
        {

        }


    private:
        int         N;
        complex_t   *m_coef{ nullptr };
        complex_t   *m_root{ nullptr };
    };


}