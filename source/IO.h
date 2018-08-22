#ifndef _IO_H
#define _IO_H

#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>
#include <fstream>
#include <iomanip>

class IO
{
protected:
  const std::string name_tag = "# name: " ;
  const std::string type_tag = "# type: vector" ;
  const std::string rows_tag = "# rows: 1" ;

  template <typename T>
  std::istream& load_vector_2(
    std::vector<T>&,
    const std::string&,
    std::istream&
  );
public:
    IO();
    ~IO();

    template <typename T>
    std::ostream& save_vector(
      const std::string&,
      const std::vector<T>&,
      std::ofstream&
    );

    template <typename T>
    std::vector<T> load_vector(
      const std::string&,
      std::istream&
    );

};

template <typename T>
std::istream& IO::load_vector_2(
  std::vector<T>& v,
  const std::string& name,
  std::istream& stm
){
  std::string str ;
  if ( !( std::getline( stm, str ) && str == ( name_tag + name ) ) ){
    std::cout << "Incorrect name of data file\n";
    exit(-1);
    goto failure;
  }
  if ( !( std::getline( stm, str ) && str == type_tag )            ) goto failure;
  if ( !( std::getline( stm, str ) && str == rows_tag )            ) goto failure;

  char ch;
  std::size_t expected_size ;
  if( !( stm >> ch && ch == '#' && stm >> str && str == "columns:" && stm >> expected_size ) )
      goto failure ;

  v.clear() ;
  using iterator = std::istream_iterator<T> ;
  std::copy( iterator(stm), iterator(), std::back_inserter(v) ) ;
  if( v.size() != expected_size ) goto failure ;

  return stm ;

  failure:
      stm.setstate( std::ios::failbit ) ;
      v.clear() ;
      return stm ;
}

template <typename T>
std::ostream& IO::save_vector(
  const std::string& name,
  const std::vector<T>& v,
  std::ofstream& stm
){
  stm << name_tag << name << '\n' << type_tag << '\n' << rows_tag << '\n'
	    << "# columns: " << v.size() << '\n' ;

  stm.setf(std::ios_base::showpos);
  stm.precision(20);

  int i = 0;
  for( auto it = v.begin(); it != v.end(); it ++ ){
    stm << std::left << std::fixed << *it << " ";
    //std::copy( it, it + 5, std::ostream_iterator<T>( stm, " " ) ) ;
    i++;
    if( i == 5 ){
      stm << "\n";
      i = 0;
    }
  }
	return stm << "\n\n\n" ;
}

template <typename T>
std::vector<T> IO::load_vector(
  const std::string& name,
  std::istream& stm
){
  std::vector<T> v;
  if( !load_vector_2( v, name, stm ).eof() ) std::cerr << "input failure!\n" ;
  return v;
}

#endif
