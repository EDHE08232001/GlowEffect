/*******************************************************************************************************************
* FILE NAME   :    win2main.h
*
* PROJ NAME   :    general use
*
* DESCRIPTION :    a function converts CmdLine from WinMain to argv/argc of main()
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2022 MAY 19      Yu Liu          Creation by copying from website
* https://stackoverflow.com/questions/291424/canonical-way-to-parse-the-command-line-into-arguments-in-plain-c-windows-api
*
********************************************************************************************************************/
#pragma once
#include <iostream>
#include <windows.h>
#include <shellapi.h>
#include <assert.h>


#if defined(_WIN2MAIN_1_)
inline LPSTR* CommandLineToArgvA(LPSTR lpCmdLine, INT* pNumArgs)
{
    int retval;
    retval = MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, lpCmdLine, -1, NULL, 0);
    if (!SUCCEEDED(retval))
        return NULL;

    LPWSTR lpWideCharStr = (LPWSTR)malloc(retval * sizeof(WCHAR));
    if (lpWideCharStr == NULL)
        return NULL;

    retval = MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, lpCmdLine, -1, lpWideCharStr, retval);
    if (!SUCCEEDED(retval))
    {
        free(lpWideCharStr);
        return NULL;
    }

    int numArgs;
    LPWSTR* args;
    args = CommandLineToArgvW(lpWideCharStr, &numArgs);
    free(lpWideCharStr);
    if (args == NULL)
        return NULL;

    int storage = numArgs * sizeof(LPSTR);
    for (int i = 0; i < numArgs; ++i)
    {
        BOOL lpUsedDefaultChar = FALSE;
        retval = WideCharToMultiByte(CP_ACP, 0, args[i], -1, NULL, 0, NULL, &lpUsedDefaultChar);
        if (!SUCCEEDED(retval))
        {
            LocalFree(args);
            return NULL;
        }

        storage += retval;
    }

    LPSTR* result = (LPSTR*)LocalAlloc(LMEM_FIXED, storage);
    if (result == NULL)
    {
        LocalFree(args);
        return NULL;
    }

    int bufLen = storage - numArgs * sizeof(LPSTR);
    LPSTR buffer = ((LPSTR)result) + numArgs * sizeof(LPSTR);
    for (int i = 0; i < numArgs; ++i)
    {
        assert(bufLen > 0);
        BOOL lpUsedDefaultChar = FALSE;
        retval = WideCharToMultiByte(CP_ACP, 0, args[i], -1, buffer, bufLen, NULL, &lpUsedDefaultChar);
        if (!SUCCEEDED(retval))
        {
            LocalFree(result);
            LocalFree(args);
            return NULL;
        }

        result[i] = buffer;
        buffer += retval;
        bufLen -= retval;
    }

    LocalFree(args);

    *pNumArgs = numArgs;
    return result;
}

#elif defined(_WIN2MAIN_2_)
#include <mbctype.h>

inline LPSTR* CommandLineToArgvA(_In_opt_ LPCSTR lpCmdLine, _Out_ int* pNumArgs)
{
    if (!pNumArgs)
    {
        SetLastError(ERROR_INVALID_PARAMETER);
        return NULL;
    }
    *pNumArgs = 0;
    /*follow CommandLinetoArgvW and if lpCmdLine is NULL return the path to the executable.
    Use 'programname' so that we don't have to allocate MAX_PATH * sizeof(CHAR) for argv
    every time. Since this is ANSI the return can't be greater than MAX_PATH (260
    characters)*/
    CHAR programname[MAX_PATH] = {};
    /*pnlength = the length of the string that is copied to the buffer, in characters, not
    including the terminating null character*/
    DWORD pnlength = GetModuleFileNameA(NULL, programname, MAX_PATH);
    if (pnlength == 0) //error getting program name
    {
        //GetModuleFileNameA will SetLastError
        return NULL;
    }
    if (*lpCmdLine == NULL)
    {

        /*In keeping with CommandLineToArgvW the caller should make a single call to HeapFree
        to release the memory of argv. Allocate a single block of memory with space for two
        pointers (representing argv[0] and argv[1]). argv[0] will contain a pointer to argv+2
        where the actual program name will be stored. argv[1] will be nullptr per the C++
        specifications for argv. Hence space required is the size of a LPSTR (char*) multiplied
        by 2 [pointers] + the length of the program name (+1 for null terminating character)
        multiplied by the sizeof CHAR. HeapAlloc is called with HEAP_GENERATE_EXCEPTIONS flag,
        so if there is a failure on allocating memory an exception will be generated.*/
        LPSTR* argv = static_cast<LPSTR*>(HeapAlloc(GetProcessHeap(),
            HEAP_ZERO_MEMORY | HEAP_GENERATE_EXCEPTIONS,
            (sizeof(LPSTR) * 2) + ((pnlength + 1) * sizeof(CHAR))));
        memcpy(argv + 2, programname, pnlength + 1); //add 1 for the terminating null character
        argv[0] = reinterpret_cast<LPSTR>(argv + 2);
        argv[1] = nullptr;
        *pNumArgs = 1;
        return argv;
    }
    /*We need to determine the number of arguments and the number of characters so that the
    proper amount of memory can be allocated for argv. Our argument count starts at 1 as the
    first "argument" is the program name even if there are no other arguments per specs.*/
    int argc = 1;
    int numchars = 0;
    LPCSTR templpcl = lpCmdLine;
    bool in_quotes = false;  //'in quotes' mode is off (false) or on (true)
    /*first scan the program name and copy it. The handling is much simpler than for other
    arguments. Basically, whatever lies between the leading double-quote and next one, or a
    terminal null character is simply accepted. Fancier handling is not required because the
    program name must be a legal NTFS/HPFS file name. Note that the double-quote characters are
    not copied.*/
    do {
        if (*templpcl == '"')
        {
            //don't add " to character count
            in_quotes = !in_quotes;
            templpcl++; //move to next character
            continue;
        }
        ++numchars; //count character
        templpcl++; //move to next character
        if (_ismbblead(*templpcl) != 0) //handle MBCS
        {
            ++numchars;
            templpcl++; //skip over trail byte
        }
    } while (*templpcl != '\0' && (in_quotes || (*templpcl != ' ' && *templpcl != '\t')));
    //parsed first argument
    if (*templpcl == '\0')
    {
        /*no more arguments, rewind and the next for statement will handle*/
        templpcl--;
    }
    //loop through the remaining arguments
    int slashcount = 0; //count of backslashes
    bool countorcopychar = true; //count the character or not
    for (;;)
    {
        if (*templpcl)
        {
            //next argument begins with next non-whitespace character
            while (*templpcl == ' ' || *templpcl == '\t')
                ++templpcl;
        }
        if (*templpcl == '\0')
            break; //end of arguments

        ++argc; //next argument - increment argument count
        //loop through this argument
        for (;;)
        {
            /*Rules:
              2N     backslashes   + " ==> N backslashes and begin/end quote
              2N + 1 backslashes   + " ==> N backslashes + literal "
              N      backslashes       ==> N backslashes*/
            slashcount = 0;
            countorcopychar = true;
            while (*templpcl == '\\')
            {
                //count the number of backslashes for use below
                ++templpcl;
                ++slashcount;
            }
            if (*templpcl == '"')
            {
                //if 2N backslashes before, start/end quote, otherwise count.
                if (slashcount % 2 == 0) //even number of backslashes
                {
                    if (in_quotes && *(templpcl + 1) == '"')
                    {
                        in_quotes = !in_quotes; //NB: parse_cmdline omits this line
                        templpcl++; //double quote inside quoted string
                    }
                    else
                    {
                        //skip first quote character and count second
                        countorcopychar = false;
                        in_quotes = !in_quotes;
                    }
                }
                slashcount /= 2;
            }
            //count slashes
            while (slashcount--)
            {
                ++numchars;
            }
            if (*templpcl == '\0' || (!in_quotes && (*templpcl == ' ' || *templpcl == '\t')))
            {
                //at the end of the argument - break
                break;
            }
            if (countorcopychar)
            {
                if (_ismbblead(*templpcl) != 0) //should copy another character for MBCS
                {
                    ++templpcl; //skip over trail byte
                    ++numchars;
                }
                ++numchars;
            }
            ++templpcl;
        }
        //add a count for the null-terminating character
        ++numchars;
    }
    /*allocate memory for argv. Allocate a single block of memory with space for argc number of
    pointers. argv[0] will contain a pointer to argv+argc where the actual program name will be
    stored. argv[argc] will be nullptr per the C++ specifications. Hence space required is the
    size of a LPSTR (char*) multiplied by argc + 1 pointers + the number of characters counted
    above multiplied by the sizeof CHAR. HeapAlloc is called with HEAP_GENERATE_EXCEPTIONS
    flag, so if there is a failure on allocating memory an exception will be generated.*/
    LPSTR* argv = static_cast<LPSTR*>(HeapAlloc(GetProcessHeap(),
        HEAP_ZERO_MEMORY | HEAP_GENERATE_EXCEPTIONS,
        (sizeof(LPSTR) * (argc + 1)) + (numchars * sizeof(CHAR))));
    //now loop through the commandline again and split out arguments
    in_quotes = false;
    templpcl = lpCmdLine;
    argv[0] = reinterpret_cast<LPSTR>(argv + argc + 1);
    LPSTR tempargv = reinterpret_cast<LPSTR>(argv + argc + 1);
    do {
        if (*templpcl == '"')
        {
            in_quotes = !in_quotes;
            templpcl++; //move to next character
            continue;
        }
        *tempargv++ = *templpcl;
        templpcl++; //move to next character
        if (_ismbblead(*templpcl) != 0) //should copy another character for MBCS
        {
            *tempargv++ = *templpcl; //copy second byte
            templpcl++; //skip over trail byte
        }
    } while (*templpcl != '\0' && (in_quotes || (*templpcl != ' ' && *templpcl != '\t')));
    //parsed first argument
    if (*templpcl == '\0')
    {
        //no more arguments, rewind and the next for statement will handle
        templpcl--;
    }
    else
    {
        //end of program name - add null terminator
        *tempargv = '\0';
    }
    int currentarg = 1;
    argv[currentarg] = ++tempargv;
    //loop through the remaining arguments
    slashcount = 0; //count of backslashes
    countorcopychar = true; //count the character or not
    for (;;)
    {
        if (*templpcl)
        {
            //next argument begins with next non-whitespace character
            while (*templpcl == ' ' || *templpcl == '\t')
                ++templpcl;
        }
        if (*templpcl == '\0')
            break; //end of arguments
        argv[currentarg] = ++tempargv; //copy address of this argument string
        //next argument - loop through it's characters
        for (;;)
        {
            /*Rules:
              2N     backslashes   + " ==> N backslashes and begin/end quote
              2N + 1 backslashes   + " ==> N backslashes + literal "
              N      backslashes       ==> N backslashes*/
            slashcount = 0;
            countorcopychar = true;
            while (*templpcl == '\\')
            {
                //count the number of backslashes for use below
                ++templpcl;
                ++slashcount;
            }
            if (*templpcl == '"')
            {
                //if 2N backslashes before, start/end quote, otherwise copy literally.
                if (slashcount % 2 == 0) //even number of backslashes
                {
                    if (in_quotes && *(templpcl + 1) == '"')
                    {
                        in_quotes = !in_quotes; //NB: parse_cmdline omits this line
                        templpcl++; //double quote inside quoted string
                    }
                    else
                    {
                        //skip first quote character and count second
                        countorcopychar = false;
                        in_quotes = !in_quotes;
                    }
                }
                slashcount /= 2;
            }
            //copy slashes
            while (slashcount--)
            {
                *tempargv++ = '\\';
            }
            if (*templpcl == '\0' || (!in_quotes && (*templpcl == ' ' || *templpcl == '\t')))
            {
                //at the end of the argument - break
                break;
            }
            if (countorcopychar)
            {
                *tempargv++ = *templpcl;
                if (_ismbblead(*templpcl) != 0) //should copy another character for MBCS
                {
                    ++templpcl; //skip over trail byte
                    *tempargv++ = *templpcl;
                }
            }
            ++templpcl;
        }
        //null-terminate the argument
        *tempargv = '\0';
        ++currentarg;
    }
    argv[argc] = nullptr;
    *pNumArgs = argc;
    return argv;
}

#elif defined(_WIN2MAIN_3_)
/*************************************************************************
 * CommandLineToArgvA            [SHELL32.@]
 *
 * MODIFIED FROM https://www.winehq.org/ project
 * We must interpret the quotes in the command line to rebuild the argv
 * array correctly:
 * - arguments are separated by spaces or tabs
 * - quotes serve as optional argument delimiters
 *   '"a b"'   -> 'a b'
 * - escaped quotes must be converted back to '"'
 *   '\"'      -> '"'
 * - consecutive backslashes preceding a quote see their number halved with
 *   the remainder escaping the quote:
 *   2n   backslashes + quote -> n backslashes + quote as an argument delimiter
 *   2n+1 backslashes + quote -> n backslashes + literal quote
 * - backslashes that are not followed by a quote are copied literally:
 *   'a\b'     -> 'a\b'
 *   'a\\b'    -> 'a\\b'
 * - in quoted strings, consecutive quotes see their number divided by three
 *   with the remainder modulo 3 deciding whether to close the string or not.
 *   Note that the opening quote must be counted in the consecutive quotes,
 *   that's the (1+) below:
 *   (1+) 3n   quotes -> n quotes
 *   (1+) 3n+1 quotes -> n quotes plus closes the quoted string
 *   (1+) 3n+2 quotes -> n+1 quotes plus closes the quoted string
 * - in unquoted strings, the first quote opens the quoted string and the
 *   remaining consecutive quotes follow the above rule.
 */

inline LPSTR* WINAPI CommandLineToArgvA(LPSTR lpCmdline, int* numargs)
{
    DWORD argc;
    LPSTR* argv;
    LPSTR s;
    LPSTR d;
    LPSTR cmdline;
    int qcount, bcount;

    if (!numargs || *lpCmdline == 0)
    {
        SetLastError(ERROR_INVALID_PARAMETER);
        return NULL;
    }

    /* --- First count the arguments */
    argc = 1;
    s = lpCmdline;
    /* The first argument, the executable path, follows special rules */
    if (*s == '"')
    {
        /* The executable path ends at the next quote, no matter what */
        s++;
        while (*s)
            if (*s++ == '"')
                break;
    }
    else
    {
        /* The executable path ends at the next space, no matter what */
        while (*s && *s != ' ' && *s != '\t')
            s++;
    }
    /* skip to the first argument, if any */
    while (*s == ' ' || *s == '\t')
        s++;
    if (*s)
        argc++;

    /* Analyze the remaining arguments */
    qcount = bcount = 0;
    while (*s)
    {
        if ((*s == ' ' || *s == '\t') && qcount == 0)
        {
            /* skip to the next argument and count it if any */
            while (*s == ' ' || *s == '\t')
                s++;
            if (*s)
                argc++;
            bcount = 0;
        }
        else if (*s == '\\')
        {
            /* '\', count them */
            bcount++;
            s++;
        }
        else if (*s == '"')
        {
            /* '"' */
            if ((bcount & 1) == 0)
                qcount++; /* unescaped '"' */
            s++;
            bcount = 0;
            /* consecutive quotes, see comment in copying code below */
            while (*s == '"')
            {
                qcount++;
                s++;
            }
            qcount = qcount % 3;
            if (qcount == 2)
                qcount = 0;
        }
        else
        {
            /* a regular character */
            bcount = 0;
            s++;
        }
    }

    /* Allocate in a single lump, the string array, and the strings that go
     * with it. This way the caller can make a single LocalFree() call to free
     * both, as per MSDN.
     */
    argv = (LPSTR*)LocalAlloc(LMEM_FIXED, (argc + 1) * sizeof(LPSTR) + (strlen(lpCmdline) + 1) * sizeof(char));
    if (!argv)
        return NULL;
    cmdline = (LPSTR)(argv + argc + 1);
    strcpy(cmdline, lpCmdline);

    /* --- Then split and copy the arguments */
    argv[0] = d = cmdline;
    argc = 1;
    /* The first argument, the executable path, follows special rules */
    if (*d == '"')
    {
        /* The executable path ends at the next quote, no matter what */
        s = d + 1;
        while (*s)
        {
            if (*s == '"')
            {
                s++;
                break;
            }
            *d++ = *s++;
        }
    }
    else
    {
        /* The executable path ends at the next space, no matter what */
        while (*d && *d != ' ' && *d != '\t')
            d++;
        s = d;
        if (*s)
            s++;
    }
    /* close the executable path */
    *d++ = 0;
    /* skip to the first argument and initialize it if any */
    while (*s == ' ' || *s == '\t')
        s++;
    if (!*s)
    {
        /* There are no parameters so we are all done */
        argv[argc] = NULL;
        *numargs = argc;
        return argv;
    }

    /* Split and copy the remaining arguments */
    argv[argc++] = d;
    qcount = bcount = 0;
    while (*s)
    {
        if ((*s == ' ' || *s == '\t') && qcount == 0)
        {
            /* close the argument */
            *d++ = 0;
            bcount = 0;

            /* skip to the next one and initialize it if any */
            do {
                s++;
            } while (*s == ' ' || *s == '\t');
            if (*s)
                argv[argc++] = d;
        }
        else if (*s == '\\')
        {
            *d++ = *s++;
            bcount++;
        }
        else if (*s == '"')
        {
            if ((bcount & 1) == 0)
            {
                /* Preceded by an even number of '\', this is half that
                 * number of '\', plus a quote which we erase.
                 */
                d -= bcount / 2;
                qcount++;
            }
            else
            {
                /* Preceded by an odd number of '\', this is half that
                 * number of '\' followed by a '"'
                 */
                d = d - bcount / 2 - 1;
                *d++ = '"';
            }
            s++;
            bcount = 0;
            /* Now count the number of consecutive quotes. Note that qcount
             * already takes into account the opening quote if any, as well as
             * the quote that lead us here.
             */
            while (*s == '"')
            {
                if (++qcount == 3)
                {
                    *d++ = '"';
                    qcount = 0;
                }
                s++;
            }
            if (qcount == 2)
                qcount = 0;
        }
        else
        {
            /* a regular character */
            *d++ = *s++;
            bcount = 0;
        }
    }
    *d = '\0';
    argv[argc] = NULL;
    *numargs = argc;

    return argv;
}

#endif