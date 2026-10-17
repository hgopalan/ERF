#ifndef ERF_OPENFAST_MODEL_FILES_H
#define ERF_OPENFAST_MODEL_FILES_H

/** OpenFAST input files read once each, however many values the start-up checks look up in them. */

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace erf_openfast {

/**
 * The lines of the OpenFAST input files a set of checks reads, each file read at its first use and
 * kept: the readers in ERF_OpenFASTAudit.H and the checks in ERF_OpenFASTDriver.H that take one look
 * up every value in the kept lines, as they would in the file. A file that cannot be opened has no
 * lines. The files are taken as unchanged while the object lives; make one per set of checks.
 */
class ModelFiles
{
public:
    const std::vector<std::string>& lines (const std::string& fname)
    {
        auto it = m_lines.find(fname);
        if (it != m_lines.end()) { return it->second; }
        std::vector<std::string> v;
        std::ifstream in(fname);
        std::string line;
        while (std::getline(in, line)) { v.push_back(line); }
        ++m_reads;
        return m_lines.emplace(fname, std::move(v)).first->second;
    }

    /** How many files were opened (once each). */
    int num_reads () const { return m_reads; }

private:
    std::map<std::string, std::vector<std::string>> m_lines;
    int m_reads{0};
};

} // namespace erf_openfast

#endif
