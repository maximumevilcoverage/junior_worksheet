#include <ctime>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <utility>
#include <numeric>
#include <iterator>
#include <functional>
#include <cctype>
#include <cstdarg>
#include <deque>
#include <set>
#include <unordered_set>
#include <complex>
#include <cassert>
using std::map;
using std::set;
using std::unordered_set;
using std::deque;
using std::cout;
using std::cin;
using std::sort; //from algorithm
using std::getline; //from string
using std::isalnum; //is alphanumeric
using std::isalpha; //is alphabetic
using std::make_pair;
using std::vector;
using std::ostream;
using std::swap;
using std::string;
using std::complex;
template <typename... Args>
auto gl(Args&&... args) -> decltype(getline(std::forward<Args>(args)...)) {
  return getline(std::forward<Args>(args)...);
}
//using std::gcd; //from numeric, C++17 only
//using std::copy; //from algorithm
// Shortcuts for "common" data types in contests
typedef unordered_set<char> sc;
typedef unordered_set<int> si;
typedef std::string str;
typedef long long ll;
typedef unsigned long ul;
typedef unsigned long long ull;
typedef std::pair<int, int> pi;
typedef std::vector<pi> vpi;
typedef std::vector<int> vi;
typedef std::vector<vi> vvi;
typedef std::vector<char> vc;
typedef std::vector<str> vs;
typedef std::map<str,int> msi;
typedef std::map<int,int> mii;
#define MAP(v,op) std::transform(v.begin(), v.end(), v.begin(), op)
//usage: MAP(s, [](char c){return toupper(c);});
#define FOLD(v,op) std::accumulate(v.begin()+1, v.end(), v[0], op) //using FOLD on empty lists is UB!!!
//usage: FOLD(v, [](int result_so_far, int next){return result_so_far + next;});
#define STR(v) std::to_string(v)
#define LOG( msg )  std::cout << __FILE__ << ":" << __LINE__ << ": " << msg
#define PV(v) printv(v,#v,__LINE__)
#define SUM(v) std::accumulate(v.begin(), v.end(), 0) //return type is same as type of init
#define PROD(v) std::accumulate(begin(v), end(v), 1, std::multiplies<>()) // product of the elements
#define XOR(v) std::accumulate(begin(v), end(v), 0, std::bit_xor<>()) //defined in functional
#define MIN(v) (*std::min_element( std::begin(v), std::end(v) ))
#define MIN_INDEX(v) (std::min_element( std::begin(v), std::end(v) ) - v.begin())
#define MAX(v) (*std::max_element( std::begin(v), std::end(v) ))
#define MAX_INDEX(v) (std::max_element( std::begin(v), std::end(v) ) - v.begin())
#define HAS(c,x) ((c).find(x) != (c).end())
#define RSUB(v) std::accumulate(rbegin(v)+1, rend(v), v.back(), std::minus<>()) //minus from right to left
//#define RV(a) vi a; getlineintovi(a) //needs getlineintovi defined
#define INPUT(s, ...) do { std::istringstream iss(s); input(iss, __VA_ARGS__); } while (0)
//usage: INPUT(s, a, b, c)
//the GLI macro reads a line into a variadic list of ints (or something else). You need to declare the variables yourself.
#define GL(s) str s; getline(cin,s);
#define GLI(...) do {str s;getline(cin,s);INPUT(s,__VA_ARGS__);} while(0) //needs the input variadic templates defined
#define GLL(...) ll __VA_ARGS__; gli(__VA_ARGS__);
#define GLUL(var) ul var; gli(var);
#define GLSTR(var) str var; gli(var);
//the SV macro converts a line into a vector of ints. It declares the vector for you.
/*
 * Note: SV(s,v) creates a new vector v out of s
 * Whereas s2v(s,v) copies s into an existing vector v
 * To use s2v you must first create a vector - opposite for SV
 * Use of SV may lead to subtle bugs
 * Use s2v unless you know what you're doing
*/
#define SVI(s,v) vi v{std::istream_iterator<int>{std::istringstream(s) >> std::skipws}, std::istream_iterator<int>()}; //the >> skipws trick works because the extractor returns an lvalue-reference
#define VCS(v,s) std::string s(v.begin(),v.end()); //s from vc
#define SVC(s,v) vc v{s.begin(), s.end()}; //vc from s
#define INF 1000000000 // 1 billion, safer than 2B for Floyd Warshall’s
#define REPUL(i, a, b) \
for (ul i = ul(a); i < ul(b); ++i)
#define REP(i, begin, end) for (__typeof(end) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); i += 1 - 2 * ((begin) > (end)))
#define L(b) \
for (ul TEMP_ = 0; TEMP_ < ul(b); ++TEMP_)
#define L2(b) \
for (ul TEMP__ = 0; TEMP__ < ul(b); ++TEMP__)
#define LR(i,b) \
for (ll i = 0; i < ll(b); ++i)
#define LR1(i,b) \
for (ll i = 1; i <= ll(b); ++i)
#define LRR(i,b) \
for (size_t i = ul(b); i --> 0;)
#define LOOPREAD(c, terminator) while(cin.get(c)){ if (c == terminator) break; //hideously disgusting macro, avoid if possible.
#define BY(x) [](const auto& a, const auto& b) { return a.x < b.x; }
//usage: sort(arr, arr + N, BY(a)); //where arr is an array of objects each containing a field named x
#define F first
#define S second
#define B(x) std::begin(x)
#define E(x) std::end(x)
#define PB push_back
#define PF push_front
#define MP make_pair
#define SZ(a) int((a).size())
#define ODD(num) (num & 1)
#define ISPRF(prefix,s) (s.compare(0, prefix.size(), prefix) == 0)
#define XSWAP(a,b) (a)^=(b); (b)^=(a); (a)^=(b); //xor swap
#define REV(s) std::reverse(begin(s), end(s));
#define ALL(a) a.begin(), a.end()
#define IN(a,b)  ( (b).find(a) != (b).end())
#define BITSET(n,b)   ( (n >> b) & 1)
// Useful hardware instructions
#define POPCNT __builtin_popcount
#define GCD __gcd
#define DIV(top,bot,q,r) int q = top / bot; int r = top % bot;
//auto [q,r] = div(gift,num); //If only judges supported C++17 :(((
/*
 * The MEMO macro creates a memoized function which relies on a helper function
 * When the memoized function doesn't have the result stored in cache it calls the helper function
 * The helper function can call back to the memoized function with a reduced parameter,
 * and the memoized function will simply pass that parameter straight back to the helper function.
 *
 * Currently MEMO only supports one parameter, and this parameter is bound to the name "x" in the function.
 *
 * Example usage:
    MEMO(fib,int,int){
        if (x == 0){return 0;}
        if (x == 1 or x == 2){return 1;}
        return (fib(x-1)%1000000007) + (fib(x-2)%1000000007);
    }
 * */
#define MEMO(funcname, parametertype, returntype) \
returntype funcname##_(parametertype); \
returntype funcname(parametertype key){ \
static std::map<parametertype,returntype> cache; \
auto it = cache.find(key); \
if (it != end(cache)){return it->second;} \
auto value = funcname##_(key); \
return cache.emplace(key,value).first->second;} \
returntype funcname##_(parametertype x)

//ios_base::sync_with_stdio(false)

//useful functions
template <typename T, typename F>
size_t count_if(T it, F pred){
    return std::count_if(it.begin(), it.end(), pred);
}
template<typename T,typename TT> ostream& operator<<(ostream &s,std::pair<T,TT> t) {return s<<"("<<t.first<<","<<t.second<<")";}
template <typename T>
T psum(T x, T y){ return x + y; }
template <typename T>
T psub(T x, T y){ return x - y; }
template <typename T>
T pdiff(T x, T y){ return abs(x - y); }
//usage: vi v2 = zipreduce(v,v,psub,-1);
template <typename T>
T pprod(T x, T y){ return x * y; }
template <typename T>
T pdiv(T x, T y){ return x / y; }
template <typename T>
T pmax(T x, T y){ if (x > y){ return x; } else { return y; } }
template <typename T>
const deque<T>& pmax(const deque<T>& x, const deque<T>& y){
    if (x.size() > y.size()){
        return x;
    }
    return y;
}
template <typename T>
void zero(T& v){
    std::fill(v.begin(), v.end(), 0);
}
template <typename T>
void lrot(T& v, ul i){
    std::rotate(v.begin(), v.begin() + i, v.end());
}
template <typename T>
void rrot(T& v, ul i){
    std::rotate(v.rbegin(), v.rbegin() + i, v.rend());
}
template <typename T>
T pmin(T x, T y){ if (x < y){ return x; } else { return y; } }
template <typename T>
vector<T>& pmin(vector<T>& x, vector<T>& y){
    if (x.size() < y.size()){
        return x;
    }
    return y;
}
template <typename T>
T pcomp(T x, T y){ return (x==y); }

ll positive_modulo(ll i, ull n) {
    return (i % ll(n) + ll(n)) % ll(n);
}
ll ceil(ll a, ll b){
    assert(a>0 and b>0);
    return 1 + ((a - 1) / b);
}
/*
 * Modulo arithmetic class
 * */
class ModInt{
public:
    ll n=0; ull mod = 0;
    ModInt(ll num, ull modulus) : mod(modulus){
        n = positive_modulo(num,mod);
    }
    ModInt& operator=(ll i){
        n = positive_modulo(i,mod);
        return *this;
    }
    ModInt& operator++(){
        n = positive_modulo(n+1,mod);
        return *this;
    }
    ModInt& operator--(){
        n = positive_modulo(n-1,mod);
        return *this;
    }
    ModInt operator+(ll num) const {
        return ModInt(positive_modulo(n+num, mod),mod);
    }
    ModInt& operator+=(ll num) {
        n = positive_modulo(n+num, mod);
        return *this;
    }
    ModInt operator-(ll num) const {
        return ModInt(positive_modulo(n-num, mod),mod);
    }
    ModInt& operator-=(ll num) {
        n = positive_modulo(n-num, mod);
        return *this;
    }
    ModInt operator*(ll num) const {
        return ModInt(positive_modulo(n*num, mod),mod);
    }
    ModInt& operator*=(ll num) {
        n = positive_modulo(n*num, mod);
        return *this;
    }
    friend ModInt operator+ (ll num, const ModInt& m) {
        return m+num;
    }
    friend ModInt operator- (ll num, const ModInt& m) {
        return m-num;
    }
    friend ModInt operator* (ll num, const ModInt& m) {
        return m*num;
    }
    ModInt operator+(const ModInt& m) const {
        return m+n;
    }
    ModInt operator-(const ModInt& m) const {
        return m-n;
    }
    ModInt operator*(const ModInt& m) const {
        return m*n;
    }
    friend std::ostream& operator<<(std::ostream& out, const ModInt& i ){
        out << i.n;
        return out;
    }
};
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}
//we simply change cin and cout to point to files instead of having to pass ostream to the print function
//disgusting global variables, sorry.I promise I won't use this in production code.
static bool debug = false;
#ifdef ENABLE_DEBUG
static bool release = false;
#else
static bool release = true;
#endif
static std::ifstream fin;
static std::ofstream fout;
void print(pi p){
    cout << "(" << p.F << "," << p.S << ")";
}
void printtype(str){
    cout << "<string>";
}
void printtype(int){
    cout << "<int>";
}
void printtype(pi){
    cout << "<pair<int,int>>";
}
template <typename T>
void print(const vector<T> v){
    cout << "vector";
    printtype(v[0]);
    cout << " = [\n";
    for (auto e:v){
        cout << "    ";
        print(e);
        cout << '\n';
    }
    //print("]\n");
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const vector<T>& v){
    if (debug) out << "vector: [";
    char delim = debug ? ',' : ' ';
    if (v.empty()){
        out << "empty";
    }
    else{
        out << v[0];
        REP(i,1,v.size()){
            out << delim << v[i];
        }
    }
    if (debug) out << "]\n";
    return out;
}
void prt(const vpi& v){
    //print("vector<pair<int,int>>",s,"on line",STR(linenumber)+": [");
    if (v.empty()){
        cout << "empty";
    }
    else{
        bool flag = false;
        for (auto p:v){
            if (flag){
                cout << ",";
            }
            cout << "(" << p.F << "," << p.S << ")";
            flag = true;
        }
    }
    //print("]\n");
}
template <typename T, typename X>
void print(std::map<T,X> m){
    cout << "map = {\n";
    for(auto it = m.begin(); it != m.end(); ++it)
    {
        cout << "    " << it->first << " : " << it->second << "\n";
    }
    cout << "};\n";
}
template <typename Arg>
void prt(Arg&& arg)
{
    cout << std::forward<Arg>(arg);
}
template <typename Arg, typename... Args>
void prt(Arg&& arg, Args&&... args)
{
    cout << std::forward<Arg>(arg);
    cout << ' ';
    prt(std::forward<Args>(args)...);
}
template <typename... Args>
void prtl(Args&&... args) //printline
{
    prt(std::forward<Args>(args)...);
    prt('\n');
}
/*
 *
 * Big Integer class for arbitrary precision arithmetic
 *
 * Originally wrote it to hold 63 bit integers but gave up when implementing shift
 *
 * This base 10 implementation is a lot less efficient but so much easier to implement.
 *
 * */
const int big_int_base_digits = 9;
const int big_int_base = 1000000000;
class BigInt{
public:
    bool positive = true; //positive by default
    deque<long> v; //unsigned char for holding values 0-9
    BigInt(long l){ //let's do this properly
        v.PF(l);
    }
    BigInt(const str& s){ //let's do this properly
        ul groups = s.size() / big_int_base_digits;
        int r = s.size() % big_int_base_digits;
        //first read all full groups into the BigInt
        for (int i = 0; ul(i) < groups; i++){ //this is 100% correct, don't fuck with it.
            auto value = stol(s.substr((groups-i-1)*big_int_base_digits+r,big_int_base_digits));
            v.PF(value);
        }
        if (r){
            auto value = stol(s.substr(0,r));
            v.PF(value);
        }
    }
    friend ostream& operator<<(ostream &out, const BigInt &b) {
        out << b.v[0];
        if (b.v.size() > 1){
            REP(i,1,b.v.size()){
                out << std::setw(big_int_base_digits) << std::setfill('0') << b.v[i];
            }
        }

        return out;
    }
    void pad(ul p){
        L(p){
            v.PF(0);
        }
    }
    void truncate(){
        while(true){
            if (v.front() == 0 and v.size() > 1){
                v.pop_front();
            }
            else{
                break;
            }
        }
    }
    BigInt& operator<<=(ul i){ // don't change this, it is correct
        std::rotate(v.begin(), v.begin() + i, v.end());
        return *this;
    }
    BigInt& operator+=(BigInt bi){
        if (v.size() < bi.v.size()){
            pad(bi.v.size() - v.size());
        }
        else if (v.size() > bi.v.size()){
            bi.pad(v.size() - bi.v.size());
        }
        bool carry = 0; long result;
        for(long i = v.size()-1; i >= 0; i--){
            result = v[i] + bi.v[i] + carry;
            carry = result / big_int_base; //see if highest bit is set
            result %= big_int_base; //unset highest bit
            v[i] = result;
        }
        if (carry){
            v.PF(1);
        }
        return *this;
    }
    BigInt operator*(const BigInt& b){
        ull carry = 0; ull result = 0;
        BigInt runningsum{"0"};
        BigInt temp{"0"};
        ul templen = v.size() + b.v.size();
        temp.v.resize(templen);
        auto diff = templen - b.v.size();
        int j=0; int pos = 0;
        LRR(i,v.size()){ //multiply each digit in v with b
            zero(temp.v); carry = 0;
            LRR(j,b.v.size()){ //each digit of b
                result = v[i] * b.v[j] + carry;
                carry = result / big_int_base;
                result %= big_int_base;
                temp.v[diff+j] = result;
            }
            if (carry){
                temp.v[diff+j-1] = carry;
            }
            temp <<= pos; pos++;
            runningsum += temp;
        }
        runningsum.truncate();
        return runningsum;
    }
    BigInt& operator*=(const BigInt& b){
        BigInt result = *this * b;
        positive = result.positive;
        v = std::move(result.v); //steal v from result
        return *this;
    }
};
/*
 * Matrix class todo: Make the multiplicand matrix column-major for better CPU caching behavior.
 *
 * */
class Mat {
public:
    vector<ll> v;
    ul rows, cols; bool mod = false; ul n = 0;
    Mat(ul rows, ul cols):rows(rows), cols(cols){
        v.resize(rows * cols);
    }
    Mat(ul rows, ul cols, ul n):rows(rows), cols(cols), n(n){
        if (n != 0){
            mod = true;
        }
        v.resize(rows * cols);
    }
    ll dot(const Mat& a, const Mat& b, ul arow, ul bcol){
        ll result = 0;
        if (mod){
            LR(i,a.cols){
                result = (result + (a.v[arow*a.cols+i] * b.v[b.cols*i+bcol]) % n) % n;
            }
        }
        else{
            LR(i,a.cols){
                result += a.v[arow*a.cols+i] * b.v[b.cols*i+bcol];
            }
        }
        return result;
    }
    Mat& inr(const vector<ll>& lls){ //insert row into matrix
        assert(lls.size() == cols && "Inserted row is wrong length");
        rows++;
        for (auto l:lls){
            v.PB(l);
        }
        return *this;
    }
    Mat& inr(ul row, const vector<ll>& lls){ //insert row into matrix
        assert(lls.size() == cols && "Inserted row is wrong length");
        assert(row < rows && "Specified row number is out of bounds");
        ul start = row * cols;
        ul end = (row+1)*cols;
        REP(i, start, end){
            v[i] = lls[i - start];
        }
        return *this;
    }
    ll& operator()(ul i, ul j){
        return v[i*cols+j];
    }
    Mat& operator+=(const Mat& m){
        assert(rows == m.rows && cols == m.cols && "Can't add 2 matrices of different size");
        if (mod) {
            LR(i,v.size()){
                v[i] = (v[i] + m.v[i]) % n;
            }
        }
        else{
            LR(i,v.size()){
                v[i] += m.v[i];
            }
        }
        return *this;
    }
    Mat& operator-=(const Mat& m){
        assert(rows == m.rows && cols == m.cols && "Can't subtract 2 matrices of different size");
        LR(i,v.size()){
            v[i] -= m.v[i];
        }
        return *this;
    }
    void mult(Mat& result, const Mat& m){
        LR(i,rows){
            LR(j,m.cols){
                result(i,j) = dot(*this,m,i,j);
            }
        }
    }
    Mat operator*(const Mat& m){
        assert(cols == m.rows && "Matrix multiplication requires A.cols = B.rows");
        Mat result(rows,m.cols, n);
        mult(result, m);
        return result;
    }
    Mat& operator*=(const Mat& m){
        assert(cols == m.rows && "Matrix multiplication requires A.cols = B.rows");
        Mat result(rows,m.cols);
        mult(result, m);
        rows = result.rows;
        cols = result.cols;
        v = std::move(result.v); //move the internal vector of the result matrix to this matrix.
        return *this;
    }
};
std::ostream& operator<<(std::ostream &out, const Mat& m){
    out << "Matrix: {\n";
    LR(i,m.rows){
        out << "   ";
        LR(j,m.cols){
            out << " " << m.v[i*m.cols + j];
        }
        out << '\n';
    }
    return out << "}\n";
}

template <typename T>
void input(std::istringstream& iss, T& arg)
{
    iss >> arg;
}
template <typename T>
void input(std::istringstream& iss, vector<T>& v) //thank god for overload resolution
{
    v.clear();
    std::copy(std::istream_iterator<T>(iss), //s/iss/cin for reading from stdin
            std::istream_iterator<T>(),
            std::back_inserter(v));
}
template <typename T, typename... Args>
void input(std::istringstream& iss,T& arg, Args&... args) //recursive variadic function
{
    iss >> arg;
    input(iss, args...);
}
//convert a string of ints to a vector of ints
/*
void s2v(const str& s, vi& v){
    v.clear();
    std::istringstream iss(s);
    std::copy(std::istream_iterator<int>(iss), //s/iss/cin for reading from stdin
            std::istream_iterator<int>(),
            std::back_inserter(v));
}*/
template <typename T>
void s2v(const str& s, vector<T>& v){
    v.clear();
    std::istringstream iss(s);
    std::copy(std::istream_iterator<T>(iss), //s/iss/cin for reading from stdin
            std::istream_iterator<T>(),
            std::back_inserter(v));
}
void getlineintovi(vi& v){
    std::string line;
    getline(cin, line);
    s2v(line,v);
}
char change_case (char c) {
    if (std::isupper(c))
        return char(std::tolower(c));
    else
        return char(std::toupper(c));
}
void strip_nonalnum(str& s){ //strip non-alphanumeric
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return !isalnum(c); }), s.end());
    //std::not1(std::ptr_fun( (int(*)(int))std::isalnum ))
}
long mismatch_index(const str& s1, const str& s2){
    auto p = std::mismatch(std::begin(s1),std::end(s1),std::begin(s2),std::end(s2));
    auto it = p.F;
    return distance(std::begin(s1), it);
}
//todo: make a variadic template version of this
vpi zip(const vi& a, const vi& b){ //requires a and b to be same length
    vpi result;
    if (a.size() != b.size()){LOG("size of a and b not equal");}
    LR(i,a.size()){
        result.emplace_back(a[i],b[i]);
    }
    return result;
}
//use boost's implementation of zipreduce instead
template <typename T>
vector<T> zipreduce(const vector<T>& a, const vector<T>& b, int d, T (op)(T,T)){ //add more default parameters as needed.
    if (a.size() != b.size()){LOG("size of a and b not equal");}
    vector<T> result; ul displacement = ul(d); bool flag = true; T arg1, arg2;
    if (d < 0) { displacement = ul(-d); flag = false;}
    LR(i,a.size()-displacement){
        if (flag){ arg1 = a[i]; arg2 = b[i+displacement]; }
        else{ arg1 = a[i+displacement]; arg2 = b[i]; }
        result.emplace_back(op(arg1,arg2));
    }
    return result;
}
//usage: vi v = zipreduce(a,b,[](int a, int b){return a+b;});
template <typename T>
void esort(T& v){ //easy sort
    sort(v.begin(),v.end());
}
template <typename T>
void rsort(T& v){ //reverse easy sort
    sort(v.rbegin(),v.rend());
}
template <typename T, typename V>
long efind(T begin, T end, V value){ //easy find
    return std::find(begin,end,value)-begin;
}
template <typename T, typename V>
long efind(T v, V value){ //easy find
    return std::find(B(v),E(v),value)-B(v);
}
vi lfreq(const str& s){ //letter frequencies
    vi f(26);
    for (char c:s){
        f[ul(c-'a')]++;
    }
    return f;
}
vi nfreq(const vi& v){ //number frequencies
    int max = v[0];
    for (int i:v){
        if (i<0){ LOG("Error: vector contains negative numbers, use map instead?"); }
        max = pmax(max,i);
        if (max > 1000000) { LOG("Error: max of vector larger than 1 million"); }
    }
    vi result(static_cast<ul>(max+1));
    for (int i:v){
        result[static_cast<ul>(i)]++;
    }
    return result;
}
template <typename T>
std::map<T,ul> freq(vector<T> v){
    std::map<T,ul> m;
    for (auto t:v){
        m[t]++;
    }
    return m;
}
template <typename K, typename V>
class OrderedMap {
public:
    std::map<K,V> m;
    vector<K> v;
    OrderedMap& add(const K& k, V v){
        this->v.push_back(k);
        m[k] = v;
        return *this;
    }
    V& operator[](const K& k){
        return m[k];
    }
    void prt(){
        for (const K& k:v){
            cout << k << ' ' << m[k] << '\n';
        }
    }
};
typedef OrderedMap<str,int> omsi;
int max(vi v){ //no point in using this. Just use MAX instead.
    return MAX(v);
}
template <typename K, typename V>
std::pair<K,V> max(std::map<K,V> m){ //find max entry in map by value, MAX finds by key
    return *std::max_element(std::begin(m), std::end(m),
                [] (const std::pair<K,V> & p1, const std::pair<K,V> & p2) {
                    return p1.second < p2.second;
                });
}
template <typename K, typename V>
std::pair<K,V> min(std::map<K,V> m){ //find min entry in map by value, MIN finds by key
    return *std::min_element(std::begin(m), std::end(m),
                [] (const std::pair<K,V> & p1, const std::pair<K,V> & p2) {
                    return p1.second < p2.second;
                });
}
template <typename T>
std::pair<T,ul> max(std::map<T,ul> m){
    ul max{0};
    std::pair<T,ul> result;
    for (const auto& p:m){
        if (max< p.S){
            max = p.S;
            result = p;
        }
    }
    return result;
}
template <typename... Args>
bool gli(Args&... args) //read in a variable number of variables on one line
{
    str s; bool x = false;
    if(gl(cin,s)){x = true;}
    std::istringstream iss(s);
    input(iss, args...);
    return x;
}
bool gline(str& s) //synonym for getline
{
    bool x = false;
    if(gl(cin,s)){x = true;}
    //cout << "gline: " << s << "end gline";
    return x;
}
/*
 * TODO: Make gln variadic: PipeCalcParams& set(std::initializer_list<ChParam> args)
 * */
template <typename Arg>
void gln(Arg& arg) //read in a variable number of variables, each on its own line
{
    gli(arg);
}
template <typename Arg, typename... Args>
void gln(Arg& arg, Args&... args) //read in a variable number of variables, each on its own line
{
    gli(arg);
    gln(args...);
}
vi getbits(ull x){
    vi v;
    while(x){
        v.PB(x&1);
        x >>= 1;
    }
    v.pop_back();
    REV(v);
    return v;
}
int scan (char& c){
    return scanf("%c",&c);
}
/* usage:
 *
    Mat m(0,2,1000000007);
    m.inr({1,1});
    m.inr({1,0});
    power(m,x);
    println(m(0,1));
 * */
template <typename T> //exponentiate x to the power of p
void power(T& x, ul p){
    vi v = getbits(p);
    auto id = x;
    for (int i : v){
        x *= x;
        if (i){
            x *= id;
        }
    }
}
void setio(str infile, str out=""){
    if (release){
        fin = std::ifstream(infile);
        std::cin.rdbuf(fin.rdbuf()); //redirect std::cin to in.txt!
        if (out != ""){
            fout = std::ofstream(out);
            std::cout.rdbuf(fout.rdbuf()); //redirect std::cout to out.txt!
        }
    }
}
/*
 * Date/time methods (not using <ctime>)
 *
 * */
ull num_leap_years(ull year){
    ull leapyears = year / 400;
    ull nonleaps = year / 100 - leapyears;
    leapyears = year / 4 - nonleaps;
    return leapyears;
}
ull num_leap_years_since_1900(ull year){
    return num_leap_years(year-1) - num_leap_years(1900);
}
ModInt getdayofweek(ul day, ul year){ //0 2for sunday, 6 for saturday
    ul daynum = day;
    ul yeardiff = year-1900;
    daynum += yeardiff * 365;
    daynum += num_leap_years_since_1900(year);
    ll d = daynum % 7;
    return ModInt{d,7};
}
bool isleapyear(ul year){
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    if (year % 4 == 0) return true;
    return false;
}
ul daysinyear(ul year){
    if (isleapyear((year))){ return 366;}
    return 365;
}
static vector<ul> daysinmonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static vector<ul> daysinmonthleap = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static map<ul,ul> yeardaytomonthday;
static map<ul,ul> yeardaytomonth;
static map<ul,ul> leapyeardaytomonthday;
static map<ul,ul> leapyeardaytomonth;
void compute(const vector<ul>& v, map<ul,ul>& month, map<ul,ul>& monthday){
    ul counter=1; ul i = 0; ul total = 1;
    while(i < v.size()){
        if (counter > v[i]) {i++; counter=1;}
        monthday[total] = counter;
        month[total] = i;
        counter++; total++;
    }
}
ul getdayofmonth(ul day, ul year){
    if (isleapyear(year)) return leapyeardaytomonthday[day];
    return yeardaytomonthday[day];
}
ul getmonthfromday(ul day, ul year){
    if (isleapyear(year)) return leapyeardaytomonth[day];
    return yeardaytomonth[day];
}
/* usage:
   compute(daysinmonth, yeardaytomonthday);
   compute(daysinmonthleap, leapyeardaytomonthday);
* */
//useless BigInt factorial function lol
BigInt factorial(ll n){
    BigInt result{"1"};
    REP(i,1,n+1){
        result *= i;
    }
    return result;
}
ModInt factorial(ll n, ull mod){
    ModInt result(1,mod);
    REP(i,1,n+1){
        result *= i;
    }
    return result;
}
int clamp(int inp, int max, int min){
    if (inp > max) return max;
    if (inp < min) return min;
    return inp;
}
int wheeldiff(int a, int b, int total){ //as per CF731-D2-A
    /* old code
    int half = total / 2;
    int diff = abs(a-b);
    if (diff > half){
        return total - diff;
    }
    return diff; */
    return pmin(abs(a-b),total-abs(a-b));
}
template <typename T>
int num_unique(T l){ //gets number of elements remaining after removing all consecutive duplicates in list
    auto it = std::unique(l.begin(), l.end());
    return std::distance(it, l.end());
}
template <typename T>
size_t count_unique(T l){ //gets number of elements remaining after removing all consecutive duplicates in list
    std::unordered_set<typename T::value_type> s;
    for (auto e:l){
        s.insert(e);
    }
    return s.size();
}
ll gcd(ll a, ll b){
    ll t;
    while(b){
        t = b;
        b = a%b;
        a = t;
    }
    return a;
}
const string& get_default(const map<string,string>& m, const string& key, const string& dflt){
    auto pos = m.find(key);
    return (pos != m.end() ? pos->second : dflt);
}
/*
 * Types of problem inputs
 * */
void input_format_1(){ //EOF terminates input
    int a,b; vi v;
    while(gli(a,b,v)) {
        prt(v);
        cout << SUM(v);
    }
}
void input_format_2(){ //line of 0s terminates input
    int a,b,c;
    while(gli(a,b,c), (a || b || c)){
        //print(a,b,c);
    }
}
void input_format_3(){ //integer n and k followed by n lines of input into separate vectors
    int n,k,a,b; vi v;
    gli(n,k);
    L(n){
        gli(a,b,v);
    }
}
void input_format_4(){ //integer n, integer k, followed by n lines of input into one vector
    int n; gli(n);
    str s1, s2; //2 strings representing integers, separated by a space
    L(n){
        gli(s1, s2);
    }
}
void read_individual_chars_including_newline(){
    char c;
    while(scanf("%c",&c) != EOF){
        //do stuff
    }
}
void USACO_input_format(){
    setio("ride1.in","ride1.out");
    int a, b;
    gli(a,b);
    prt(a*b);
}
/* INPUT TYPES

Input:
3 7
4 5 14

int n,h;vi a;
gli(n,h); gli(a);


Input:
3
1 1 0
1 1 1
1 0 0

int n; gli(n); vvi v;
L(n){
    vi temp;
    gli(temp);
    v.PB(temp);
}


Input:
1 5 3 2
11221


vi values; gli(values); char c;int sum=0;
LOOPREAD(c,'\n')
    sum += values[c-'1'];
}

*/

// MARK: Section name
/*
 * End of boilerplate
 * */
//release is false by default, must write release = true; to get it to be true.
int main(){
    str s; gline(s); vi v;
    for (char c:s){
        if (isdigit(c)) v.PB(c-'0');
    }
    esort(v); //could be more efficient like dutch flag but cba.
    prt(v[0]);
    REP(i,1,v.size()){
        cout<<"+"<<v[i];
    }

}
