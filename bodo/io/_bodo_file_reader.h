#ifndef _BODO_FILE_READER_H_INCLUDED
#define _BODO_FILE_READER_H_INCLUDED
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <cstdint>
#include <vector>
#include "../libs/_bodo_common.h"
#include "../libs/_distributed.h"
#include "arrow/filesystem/filesystem.h"

typedef std::vector<int64_t> size_vec;
typedef std::vector<std::string> name_vec;
typedef std::vector<boost::filesystem::path> path_vec;
typedef std::vector<std::pair<std::string, int64_t>> name_size_vec;

struct File_Type {
    enum FTypeEnum { csv = 0, json = 1, numpy = 2 };
};

#define CHECK(expr, msg)                                    \
    if (!(expr)) {                                          \
        std::cerr << "Error in read: " << msg << std::endl; \
    }

// File reader abstraction to enable plugging multiple data sources
// and file formats
class FileReader {
   public:
    File_Type::FTypeEnum f_type = File_Type::csv;
    bool json_lines = true;  // used by read_json
    FileReader(const char *f_type) { this->assign_f_type(std::string(f_type)); }
    virtual ~FileReader(){};
    virtual uint64_t getSize() = 0;
    virtual bool seek(int64_t pos) = 0;
    virtual bool ok() = 0;
    virtual bool read(char *s, int64_t size) = 0;
    /**
     * assign f_type based on file_type
     *
     * @param file_type "csv"/"json"/""
     */
    void assign_f_type(const std::string &file_type) {
        if (file_type == "csv")
            this->f_type = File_Type::csv;
        else if (file_type == "json")
            this->f_type = File_Type::json;
        else
            this->f_type = File_Type::numpy;
    };
    /**
     * f_type to string
     */
    const char *f_type_to_stirng() {
        if (this->f_type == File_Type::csv) return "csv";
        if (this->f_type == File_Type::json) return "json";
        return "";
    }
};

// File reader abstraction to enable plugging in a single file of
// multiple data sources and file formats
class SingleFileReader : public FileReader {
   public:
    const char *fname;
    SingleFileReader(const char *_fname, const char *f_type)
        : FileReader(f_type), fname(_fname){};
    virtual ~SingleFileReader(){};
    virtual uint64_t getSize() = 0;
    virtual bool seek(int64_t pos) = 0;
    virtual bool ok() = 0;
    virtual bool read_to_buff(char *s, int64_t size) = 0;
    bool read(char *s, int64_t size) {
        if (!this->read_to_buff(s, size)) return false;
        if (this->f_type == File_Type::json && (!this->json_lines)) {
            edit_json_multiline_obj(s, size);
        }
        return true;
    };

   protected:
    /**
     * manipulate json obj string so that pandas.read_json can read properly
     * when pandas.read_json(lines=True) and we already counted entries
     *
     * After manipulation, each rank should read json object that looks like:
     * [{...},\n ... ,\n{...}]\n
     * whether there is '\n' inside {} does not matter
     *
     * @param s buffer containing json obj string
     * @param size number of chars s contains
     */
    void edit_json_multiline_obj(char *s, int64_t size) {
        // change the start from '{\n' into '[{'
        if (s[0] == '{') {
            if (s[1] == '\n') {
                s[0] = '[';
                s[1] = '{';
            } else {
                Bodo_PyErr_SetString(PyExc_RuntimeError,
                                     ("Invalid Multi-Line Json format in " +
                                      std::string(this->fname) +
                                      ": '{' should be followed by '\n' at the "
                                      "beginning of the record.")
                                         .c_str());
            }
        }
        // change the end from '},\n' to '}]\n'
        if (s[size - 2] == ',' && s[size - 3] == '}') {
            if (s[size - 1] == '\n')
                s[size - 2] = ']';
            else {
                Bodo_PyErr_SetString(PyExc_RuntimeError,
                                     ("Invalid Multi-Line Json format in " +
                                      std::string(this->fname) +
                                      ": '},' should be followed by '\n' at "
                                      "the end of the record.")
                                         .c_str());
            }
        }
    }
};

// File reader abstraction to enable plugging in a directory of csv/json file
class DirectoryFileReader : public FileReader {
   public:
    const char *dirname;
    SingleFileReader *f_reader = nullptr;  // SingleFileReader for the file
                                           // currently opened/seeked/read
    int64_t file_index = 0;  // index of the file opened by f_reader
    int64_t curr_pos = 0;    // seeking position relative
                             // to the file opened by f_reader
    uint64_t dir_size = 0;   // size of all csv/json files inside the directory
                             // file_sizes[i]: cumulative size of corresponding
                             // files in file_names[0,i)
    // |file_sizes| = |files| + 1
    size_vec file_sizes;
    name_vec
        file_names;  // sorted names of each csv/json file inside the directory

    DirectoryFileReader(const char *_dirname, const char *f_type)
        : FileReader(f_type), dirname(_dirname){};
    ~DirectoryFileReader() {
        if (this->f_reader) delete this->f_reader;
    };
    virtual void initFileReader(const char *fname){};
    bool seek(int64_t pos) {
        // find which file to seek at
        this->file_index = this->findFileIndexFromPos(pos);
        // initialize S3FileReader
        initFileReader((this->file_names[this->file_index]).c_str());
        CHECK(this->ok(),
              "could not open file: " + this->file_names[this->file_index]);
        // find seeking position relative to the file, and seek into file
        this->curr_pos = this->seekIntoFileReader(pos);
        return this->ok();
    };
    bool read(char *s, int64_t size) {
        int64_t seen_size = 0;  // number of bytes seen
        int64_t read_size;      // number of bytes to read
        int64_t readable_size;  // number of bytes available for read

        while (seen_size < size) {
            readable_size = this->findReadableSize();
            read_size = this->findReadSize(size - seen_size, readable_size);

            this->f_reader->read(s + seen_size, read_size);
            CHECK(this->ok(),
                  "could not read file:" + std::string(this->f_reader->fname));

            // handle curr_pos and file_index
            if (read_size < readable_size) {
                this->curr_pos = this->curr_pos + read_size;
            } else {
                this->curr_pos = 0;
                this->file_index += 1;
                // delete the current FileReader, and initialize the next one
                if (this->file_index < (int64_t)this->file_names.size()) {
                    delete this->f_reader;
                    initFileReader(
                        (this->file_names[this->file_index]).c_str());
                    CHECK(this->ok(), "could not open file: " +
                                          this->file_names[this->file_index]);
                }
            }
            seen_size += read_size;
        }
        return this->ok();
    };
    uint64_t getSize() { return this->dir_size; };
    bool ok() {
        if (!this->f_reader) {
            return true;
        }
        return this->f_reader->ok();
    };

   protected:
    /**
     * find file index from position, byte offset in the aggregated dataset,
     * relative to the directory
     *
     * @param pos Position relative to the directory
     * @return file index of the file the position is in
     */
    int64_t findFileIndexFromPos(int64_t pos) {
        size_vec::iterator up = std::upper_bound(this->file_sizes.begin(),
                                                 this->file_sizes.end(), pos);
        return up - this->file_sizes.begin() - 1;
    };
    /**
     * seek the position in f_reader
     *
     * @param pos Position relative to the directory
     * @return Position relative to the file that is seeked
     */
    int64_t seekIntoFileReader(int64_t pos) {
        int64_t pos_in_file = pos - (this->file_sizes)[this->file_index];
        this->f_reader->seek(pos_in_file);
        return pos_in_file;
    };
    /**
     * compute dir_size, file_names, file_sizes
     *
     * dir_size, and file_sizes are set within the function
     *
     * @param dname directory name
     * @param file_names_sizes Pair(file name, size of file)
     * @return file names extracted from file_names_sizes
     */
    name_vec findDirSizeFileSizesFileNames(
        const char *dname, const name_size_vec &file_names_sizes) {
        std::string dirname(dname);
        const char *suffix;
        name_vec file_names;
        this->dir_size = 0;
        // TODO: consider the possibility of non-standard extensions
        if (this->f_type == File_Type::csv) {
            suffix = ".csv";
        } else {
            assert(this->f_type == File_Type::json);
            suffix = ".json";
        }

        for (auto it = file_names_sizes.begin(); it != file_names_sizes.end();
             ++it) {
            if ((boost::ends_with(((*it).first.c_str()), suffix))) {
                file_names.push_back((*it).first);
                (this->file_sizes).push_back(this->dir_size);
                this->dir_size += (*it).second;
            }
        }
        this->file_sizes.push_back(this->dir_size);
        return file_names;
    };
    /**
     * find size of readable contents of the file
     *
     * readable size = file size - seeked position
     *
     * @return readable size
     */
    int64_t findReadableSize() {
        // size of the file currently reading
        int64_t curr_file_size =
            file_sizes[this->file_index + 1] - file_sizes[this->file_index];
        return curr_file_size - this->curr_pos;
    };
    /**
     * find read size to pass to f_reader->read
     *
     * @param size size that still needs to be read
     *             by DirectoryFileReader->read()
     * @param readable_size size that can be read from the file seeked
     * @return read size to pass to f_reader->read
     */
    int64_t findReadSize(int64_t size, int64_t readable_size) {
        // stop reading before the end of the file
        if (size <= readable_size) {
            return size;
        }
        // read until the end of the file
        return readable_size;
    }
};

#undef CHECK
#endif  // _BODO_FILE_READER_H_INCLUDED
