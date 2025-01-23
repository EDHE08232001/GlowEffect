/*******************************************************************************************************************
* FILE NAME   :    file_util.h
*
* PROJECTION  :    general-purpose used common utility
*
* DESCRIPTION :    a file defines utility of file system
*
* VERSION HISTORY
* YYYY/MMM/DD      Author          Comments
* 2021 NOV 18      Yu Liu          Creation
*
********************************************************************************************************************/
#pragma once
#include <string>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <errno.h>
#include <io.h>

namespace file_util
{
    /// @define _S_IFMT   0xF000 // File type mask
    /// @define _S_IFDIR  0x4000 // Directory
    /// @define _S_IFCHR  0x2000 // Character special
    /// @define _S_IFIFO  0x1000 // Pipe
    /// @define _S_IFREG  0x8000 // Regular
    /// @define _S_IREAD  0x0100 // Read permission, owner
    /// @define _S_IWRITE 0x0080 // Write permission, owner
    /// @define _S_IEXEC  0x0040 // Execute/search permission, owner
    /// 
    /// dev_t     st_dev     Device ID of device containing file.
    /// ino_t     st_ino     File serial number.
    /// mode_t    st_mode    Mode of file(see below).
    /// nlink_t   st_nlink   Number of hard links to the file.
    /// uid_t     st_uid     User ID of file.
    /// gid_t     st_gid     Group ID of file.
    /// dev_t     st_rdev    Device ID(if file is character or block special).
    /// off_t     st_size    For regular files, the file size in bytes.
    ///                      For symbolic links, the length in bytes of the
    ///                      pathname contained in the symbolic link.
    ///                      For a shared memory object, the length in bytes.
    ///                      For a typed memory object, the length in bytes.
    ///                      For other file types, the use of this field is unspecified.
    /// time_t    st_atime   Time of last access.
    /// time_t    st_mtime   Time of last data modification.
    /// time_t    st_ctime   Time of last status change.

    inline bool isfile(const char* filename) {
        struct stat info;
        const int ret = stat(filename, &info);
        return ret == 0;
    }

    inline int ext_cnt(const char* filename) {
        return (int)std::string(filename).find_last_of('.') + 1;
    }

    inline int ext_chk(const char* filename, const int ext_cnt, const char* ext_ref) {
        return (int)std::string(filename).substr(ext_cnt).find(ext_ref);
        //return std::string(filename).find(ext_ref, ext_cnt);
    }

    /* file2mem(...) 
    * copy data from a file into memory
    * return true if errors occur
    */
    inline bool file2mem(const char* fn, char*& param, size_t& size)
    {
        std::ifstream   fp_par(fn);
        if (!fp_par)
            return true;
        fp_par.seekg(0, std::ios::end);
        std::streampos  par_size = fp_par.tellg();
        fp_par.seekg(0, std::ios::beg);
        param = new char[par_size];
        fp_par.read(param, par_size);
        size = (std::streamsize)par_size;
        fp_par.close();
        return false;
    }

    //========================================================
    // usage: fail if cnt < 0
    //    const int cnt = file_util::file_ext(argv[1], "ppm");
    //    
    inline int file_ext(const char* filename, const char* ext_ref) {
        return (int)ext_chk(filename, ext_cnt(filename), ext_ref);
    }

    inline const char* conv_mpg2ppm(const char* filename, const int n_files)
    {
        if (!isfile("__tmp_ppm__"))
            system("mkdir __tmp_ppm__");

        std::string opts = std::string(" -start_number 0 -vframes ") + std::to_string(n_files);
        std::string cmd = "ffmpeg -i " + std::string(filename) + opts + " ./__tmp_ppm__/frame_%04d.ppm";

        system(cmd.c_str());
        return "__tmp_ppm__/frame";
    }

    inline void clear_mpg2ppm(void)
    {
        if (isfile("__tmp_ppm__"))
            system("rmdir /s /q __tmp_ppm__");
    }

#ifdef _WIN32

    typedef int mode_t;

    /// @Note If STRICT_UGO_PERMISSIONS is not defined, then setting Read for any
    ///       of User, Group, or Other will set Read for User and setting Write
    ///       will set Write for User.  Otherwise, Read and Write for Group and
    ///       Other are ignored.
    ///
    /// @Note For the POSIX modes that do not have a Windows equivalent, the modes
    ///       defined here use the POSIX values left shifted 16 bits.

    static const mode_t S_ISUID = 0x08000000;           ///< does nothing
    static const mode_t S_ISGID = 0x04000000;           ///< does nothing
    static const mode_t S_ISVTX = 0x02000000;           ///< does nothing
    static const mode_t S_IRUSR = mode_t(_S_IREAD);     ///< read by user
    static const mode_t S_IWUSR = mode_t(_S_IWRITE);    ///< write by user
    static const mode_t S_IXUSR = 0x00400000;           ///< does nothing
#   ifndef STRICT_UGO_PERMISSIONS
    static const mode_t S_IRGRP = mode_t(_S_IREAD);     ///< read by *USER*
    static const mode_t S_IWGRP = mode_t(_S_IWRITE);    ///< write by *USER*
    static const mode_t S_IXGRP = 0x00080000;           ///< does nothing
    static const mode_t S_IROTH = mode_t(_S_IREAD);     ///< read by *USER*
    static const mode_t S_IWOTH = mode_t(_S_IWRITE);    ///< write by *USER*
    static const mode_t S_IXOTH = 0x00010000;           ///< does nothing
#   else
    static const mode_t S_IRGRP = 0x00200000;           ///< does nothing
    static const mode_t S_IWGRP = 0x00100000;           ///< does nothing
    static const mode_t S_IXGRP = 0x00080000;           ///< does nothing
    static const mode_t S_IROTH = 0x00040000;           ///< does nothing
    static const mode_t S_IWOTH = 0x00020000;           ///< does nothing
    static const mode_t S_IXOTH = 0x00010000;           ///< does nothing
#   endif
    static const mode_t MS_MODE_MASK = 0x0000ffff;           ///< low word

    static inline int chmod(const char* path, mode_t mode)
    {
        int result = _chmod(path, (mode & MS_MODE_MASK));

        if (result != 0)
        {
            result = errno;
        }

        return (result);
    }
#else
    static inline int chmod(const char* path, mode_t mode)
    {
        int result = chmod(path, mode);

        if (result != 0)
        {
            result = errno;
        }

        return (result);
    }
#endif

}