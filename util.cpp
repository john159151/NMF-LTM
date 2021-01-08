
#ifndef _UTIL_CPP
#define _UTIL_CPP


#include <iostream>
#include <string>
#include <vector>

using namespace std;

#pragma once
class util
{
public:
    util() {}
    ~util() {}
	
	string toString(double x)
	{
		string result = to_string(x);
		result.erase(result.find_last_not_of('0') + 1, string::npos); //erase字符串后面的0
		if (result[result.length() - 1] == '.') //避免出现10.的情况
			result = result.substr(0, result.length() - 1);
		return result;
	}
    
    vector<string> split(string str, string pattern)
    {
        std::string::size_type pos;
        std::vector<std::string> result;
        str+=pattern;//
        int size=str.size();

        for(int i=0; i<size; i++)
        {
            pos=str.find(pattern,i);
            if(pos<size)
            {
                std::string s=str.substr(i,pos-i);
                result.push_back(s);
                i=pos+pattern.size()-1;
            }
        }
        return result;
    }

    void test_utilsplit(){
        string str="1:2 3:4 5:6 7:8 ";
        vector<string> result=split(str," ");
        for(int i=0;i<result.size();i++){
            string s=result[i];
            cout<<"result["<<i<<"]:"<<result[i]<<"---"<<(s.compare("")==0)<<endl; //compare() 0:==; other:~0
        }
    }


private:

};

#endif
