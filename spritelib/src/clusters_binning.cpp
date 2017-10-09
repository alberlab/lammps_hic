/****************************************************************************
Usage
  ./clusters_binning BED_FILE [RAW_FILE]

Given raw clusters data and a bed domain file, writes a "cluster" file, i.e. 
uses numbering based on the domains specified in the bed file, and counts 
multiple reads from the same domain as one. If RAW_FILE is not specified,
reads from standard input. 

The chromatin segmentation bed file should be in the form "chr start end", es.
chr1          0     100000
chr1     100000     200000
....
chrX  166600000  166700000

The output is written to the standard output as text, each cluster on a line,
each line starting with the cluster size followed by the indexes of the 
domains in the cluster, es.
3  122 123 168
2  96 97
4  30 31 65 66
....

The indexes in each cluster are unique. Only clusters of size at least 2 will
be written. Reads which do not fall inside a domain defined in the BED_FILE
are ignored and discarded. The indexing of domains starts at 0 and ends at
N-1, if N domains are defined.

The raw file is expected to have a line for each cluster, in the form
barcode read_1 read_2 ... read_n. The barcode is ignored, while each read
should be on the form <chromosome>:<position>, es:
8.YbotE21.Odd2Bo60  chr1:69834191   chr6:87286853
5.YbotE72.Odd2Bo70  chr13:63805311  chr13:63438727  chr14:63010798
....
*****************************************************************************/

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <set>
#include <cmath>
#include <unordered_map>
#include <cassert>


typedef float real_t; 
typedef std::pair<int, std::vector<int> > Offsets;
typedef std::unordered_map<std::string, Offsets > Bedmap;



using namespace std;


bool is_any(const char c, const string& d)
{
  for (int i = 0; i < d.length(); ++i)
  {
    if(c == d[i]) return true;
  }
  return false;
}

vector<string> split(const string &s, const string delim = " \t\r\n") 
{
  vector<string> elems;
  const int l = s.length();

  int i = 0;
  while ( i < l && is_any(s[i],delim))
  {
    ++i;
    continue;
  }

  while (i<l)
  {
    int b = i;
    ++i;
    while (i < l && !is_any(s[i],delim))
    {
      ++i;
    }
    
    elems.push_back(s.substr(b,i-b));

    if (i == l) break;

    while (i<l && is_any(s[i],delim))
    {
      ++i;
    }
  }

  return elems;
}

Bedmap read_bed(string fname)
{
  // reads the bead files in a hash map.
  // the first element is the bin offset
  // the second is the list of starts and ends
  Bedmap map;
  ifstream inf(fname);
  string chr;
  string last_chr = "";
  int start, end;
  while (!(inf >> chr >> start >> end).fail())
  {
    if (map.count(chr) == 0)
    {
      if (last_chr == "")
      {
        map[chr].first = 0;
      }
      else
      {
        int nlast = map[last_chr].second.size() / 2;
        map[chr].first = map[last_chr].first + nlast;
      }
      last_chr = chr;
    }

    map[chr].second.push_back(start);
    map[chr].second.push_back(end);
  }

  return map;
} 

int get_bead_index(const string& s, const Bedmap& map)
{
  vector<string> vals = split(s, ":");
  string& chr = vals[0];
  int pos = atoi(vals[1].c_str());
  Bedmap::const_iterator ofs = map.find(chr);
  if (ofs == map.end())
    return -1;
  if (pos > ofs->second.second.back()){
    return -1;
  }
  int bpos = ofs->second.first; 
  int i = 0;
  // terminates when the bin end is after pos 
  while (ofs->second.second[i*2 + 1] < pos)
  {
    ++bpos;
    ++i;
  }
  // if the bin start is after pos, we fell on a gap in the bed list
  if (ofs->second.second[i*2] > pos)
  {
    return -1;
  }
  return bpos;
}





int main(int argc, char* argv[])
{
  // use with ./program input_file bed_file

  istream* input = &std::cin;
  ifstream file_input;
  if (argc < 2)
  {
    cerr << "At least one argument required\n";
    cerr << "Usage:\n  " << argv[0] << " BED_FILE [INPUT_FILE]\n";
    cerr << "If the input file is not specified, will read from";
    cerr << " standard input" << endl;
    exit(1);
  }

  // if the file is specified open it and set the input pointer 
  // to it.
  if (argc > 2)
  {
    file_input.open(argv[2], ifstream::in);
    if (not file_input.is_open())
    {
      cerr << "Failed to open file \"" << argv[2] << "\"" << endl;
      exit(1);
    }
    input = &file_input;
  }

  istream& inf = *input;



  Bedmap bed = read_bed(argv[1]);
  int tot = 0;
  for (auto& it : bed){
    tot += it.second.second.size()/2;
  }
  fprintf(stderr, "%d beads in bedfile\n", tot);
  
  string line;
  string token;

  typedef set<int> Cluster;
   
  int n_lines_read = 0; 
  while (!getline(inf, line).fail())
  {
    ++n_lines_read;
    if (n_lines_read % 100000 == 0)
    {
      fprintf(stderr, "\r%dk lines read", n_lines_read / 1000);
      fflush(stderr);
    }

    vector<string> elems = split(line);
    int n = elems.size(); 
    Cluster cluster;

    // get the bin of each element. Cluster is a set,
    // so we don't get duplicates
    for (int i = 1; i < n; ++i) // the first elements in each line is the barcode
    {
      int bead_index = get_bead_index(elems[i], bed); 
      assert(bead_index < tot);
      if (bead_index != -1)
        cluster.insert(bead_index);
    }

    n = cluster.size(); // update the real size

    if (n < 2) continue; // skip clusters with no contacts
    cout << n << ' ';

    for (int i : cluster)
    {
      cout << ' ' << i;
    }
    cout << '\n';
  }

  return 0;
}

