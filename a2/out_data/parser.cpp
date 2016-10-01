#include <bits/stdc++.h>
#define endl "\n"
#define ll long long

using namespace std;


bool startsWith(string temp, string prefix) {
    int n = prefix.length();
    for (int i = 0; i < n; ++i)
        if(temp[i] != prefix[i])
            return false;

    return true;
}

int main() {
    std::ios::sync_with_stdio(false);cin.tie(0);
    int total_models=0, candidates=0;
    string temp;
    bool fill = false;


    string data = "";
    string pre = "";

    while(getline(cin, temp)) {
        if(temp == "")
            continue;
        if(startsWith(temp, "(Iteration 150")) {
            pre = temp.substr(22, 14) + "\n";
            total_models++;
        } else if (startsWith(temp, "(Epoch 10")) {
            double val_acc = std::stod(temp.substr(46, 8));
            temp = temp.substr(16, 38);
            if (val_acc > 0.5) {
                fill = true;
                candidates++;
                data += pre + temp + "\n";
            } else {
                fill = false;
            }
        } else if (startsWith(temp, "Urule:")){
            if(fill) {
                data += temp+"\n";
                cout << data << endl;
                data = "";
            }
            fill = false;
        } else if (!startsWith(temp, "(Iteration") && !startsWith(temp, "(Epoch") && fill){
            data += temp + " ";
        }
    }
    if(fill)
        cout << data;

    cout << "Number of models:" << total_models << endl;
    cout << "Candidates:" << candidates << endl;
    return 0;
}