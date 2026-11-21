// =====================================================================
//  Water Reservoir Level Prediction System
//  ES-201 Computational Methods (PBL)            Language: C++17
//
//  What it does
//    1. Stores daily reservoir level readings (% of capacity), with
//       add / update / delete and save / load to a CSV file
//    2. Draws an ASCII graph: readings, trend line and forecast
//    3. Statistical forecast  : least-squares linear regression (+ R^2)
//    4. Physical forecast     : water balance  dV/dt = Qin(t) - Qout(V)
//                               solved with RK4 (compared with Euler)
//    5. Linear interpolation  : inflow forecast at any time
//    6. Trapezoidal rule      : total inflow volume
//    7. Error analysis        : RK4 / Euler / trapezoid vs exact answers
//
//  Compile:  g++ -std=c++17 -O2 -Wall -o reservoir Water_Reservoir_Level_Prediction.cpp
//  Run:      ./reservoir
// =====================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------
const double PI             = acos(-1.0);
const double SECONDS_PER_DAY = 86400.0;
const double LOW_LEVEL      = 25.0;   // % below this -> LOW
const double GOOD_LEVEL     = 60.0;   // % at or above this -> GOOD
const string DATA_FILE      = "reservoir_data.csv";

// ---------------------------------------------------------------------
//  Safe console input (never crashes on bad input, handles Ctrl+D)
// ---------------------------------------------------------------------
namespace input {

string trim(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

optional<string> askLine(const string& prompt) {
    cout << prompt;
    string s;
    if (!getline(cin, s)) return nullopt;          // end of input
    return trim(s);
}

optional<int> askInt(const string& prompt, int lo, int hi) {
    while (true) {
        auto line = askLine(prompt);
        if (!line) return nullopt;
        try {
            size_t pos = 0;
            int v = stoi(*line, &pos);
            if (pos == line->size() && v >= lo && v <= hi) return v;
        } catch (...) {}
        cout << "  Please enter a whole number from " << lo << " to " << hi << ".\n";
    }
}

optional<double> askDouble(const string& prompt, double lo, double hi) {
    while (true) {
        auto line = askLine(prompt);
        if (!line) return nullopt;
        try {
            size_t pos = 0;
            double v = stod(*line, &pos);
            if (pos == line->size() && isfinite(v) && v >= lo && v <= hi) return v;
        } catch (...) {}
        cout << defaultfloat << "  Please enter a number from " << lo << " to " << hi << ".\n";
    }
}

}  // namespace input

// ---------------------------------------------------------------------
//  Numerical methods
// ---------------------------------------------------------------------
namespace numeric {

typedef function<double(double, double)> OdeFunc;   // f(t, y)

// Piecewise linear interpolation of y at x, using sorted table (xs, ys)
double lerp(double x, const vector<double>& xs, const vector<double>& ys) {
    if (xs.empty()) return 0.0;
    if (x <= xs.front()) return ys.front();
    if (x >= xs.back())  return ys.back();
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        if (x <= xs[i + 1]) {
            double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    return ys.back();
}

// Composite trapezoidal rule: integral of f on [a, b] with n strips
double trapezoid(const function<double(double)>& f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i) sum += f(a + i * h);
    return sum * h;
}

// One explicit Euler step.  t, dt in days; f returns m^3/s; y in m^3
double eulerStep(const OdeFunc& f, double t, double y, double dtDays) {
    return y + dtDays * SECONDS_PER_DAY * f(t, y);
}

// One classical Runge-Kutta 4 step
double rk4Step(const OdeFunc& f, double t, double y, double dtDays) {
    double dtS = dtDays * SECONDS_PER_DAY;
    double k1 = f(t, y);
    double k2 = f(t + dtDays / 2, y + 0.5 * dtS * k1);
    double k3 = f(t + dtDays / 2, y + 0.5 * dtS * k2);
    double k4 = f(t + dtDays, y + dtS * k3);
    return y + dtS / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
}

}  // namespace numeric

// ---------------------------------------------------------------------
//  Data types
// ---------------------------------------------------------------------
struct Reading {
    int day;
    double level;    // percentage of capacity, 0..100
};

struct Regression {
    bool   valid = false;
    double slope = 0, intercept = 0, r2 = 0;
    double at(double day) const { return slope * day + intercept; }
};

// Parameters of the physical (water balance) model
struct Physics {
    double capacityMm3 = 100.0;     // full capacity, million m^3
    double demand      = 45.0;      // controlled release / water supply, m^3/s
    double lossCoeff   = 1.0e-8;    // evaporation + seepage: loss = k * V  (1/s)
    // Forecast of river inflow (m^3/s) for day 0 (today) .. day 7
    vector<double> inflowDays  = {0, 1, 2, 3, 4, 5, 6, 7};
    vector<double> inflowValue = {22, 24, 30, 38, 34, 28, 24, 22};
};

// ---------------------------------------------------------------------
//  The system
// ---------------------------------------------------------------------
class ReservoirSystem {
public:
    ReservoirSystem() { loadSample(); }

    void run() {
        while (true) {
            printMenu();
            auto choice = input::askInt("Enter your choice: ", 0, 9);
            if (!choice || *choice == 0) break;
            switch (*choice) {
                case 1: showReadings();       break;
                case 2: addOrUpdateReading(); break;
                case 3: deleteReading();      break;
                case 4: showGraph();          break;
                case 5: showTrendPrediction();break;
                case 6: showPhysicalModel();  break;
                case 7: showErrorAnalysis();  break;
                case 8: saveToFile();         break;
                case 9: loadFromFile();       break;
            }
        }
        cout << "\nThank you for using the Water Reservoir Level Prediction System!\n";
    }

private:
    vector<Reading> readings;
    Physics physics;

    // ---------------- helpers ----------------
    void loadSample() {
        readings = {{1, 72}, {2, 70}, {3, 74}, {4, 69}, {5, 67},
                    {6, 68}, {7, 64}, {8, 66}, {9, 62}, {10, 60}};
    }

    static string statusOf(double level) {
        if (level < LOW_LEVEL)  return "LOW - conserve water";
        if (level < GOOD_LEVEL) return "MODERATE";
        return "GOOD";
    }

    static void line(char c = '-', int n = 60) { cout << string(n, c) << "\n"; }

    void printMenu() const {
        cout << "\n";
        line('=');
        cout << "        WATER RESERVOIR LEVEL PREDICTION SYSTEM\n";
        line('=');
        cout << " 1. View recorded readings\n"
             << " 2. Add or update a reading\n"
             << " 3. Delete a reading\n"
             << " 4. Show ASCII graph (readings + trend + forecast)\n"
             << " 5. Predict next 7 days  (linear regression)\n"
             << " 6. Physical model       (RK4 water balance)\n"
             << " 7. Error analysis       (RK4 / Euler / trapezoid)\n"
             << " 8. Save readings to file\n"
             << " 9. Load readings from file\n"
             << " 0. Exit\n";
        line();
    }

    // Least-squares regression  y = slope * day + intercept   (+ R^2)
    Regression fitLine() const {
        Regression r;
        int n = (int)readings.size();
        if (n < 2) return r;
        double sx = 0, sy = 0, sxy = 0, sxx = 0;
        for (const auto& p : readings) {
            sx += p.day;  sy += p.level;
            sxy += p.day * p.level;  sxx += (double)p.day * p.day;
        }
        double den = n * sxx - sx * sx;
        if (fabs(den) < 1e-12) return r;
        r.slope = (n * sxy - sx * sy) / den;
        r.intercept = (sy - r.slope * sx) / n;

        double mean = sy / n, ssTot = 0, ssRes = 0;
        for (const auto& p : readings) {
            ssTot += (p.level - mean) * (p.level - mean);
            double e = p.level - r.at(p.day);
            ssRes += e * e;
        }
        r.r2 = (ssTot < 1e-12) ? 1.0 : 1.0 - ssRes / ssTot;
        r.valid = true;
        return r;
    }

    static double clampPct(double v) { return max(0.0, min(100.0, v)); }

    // ---------------- 1. view ----------------
    void showReadings() const {
        cout << "\n========== RECORDED READINGS ==========\n";
        if (readings.empty()) { cout << "No readings available.\n"; return; }
        cout << left << setw(8) << "Day" << setw(14) << "Level (%)"
             << setw(12) << "Change" << "Status\n";
        line('-', 50);
        for (size_t i = 0; i < readings.size(); ++i) {
            ostringstream ch;
            if (i == 0) ch << "-";
            else ch << showpos << fixed << setprecision(1)
                    << readings[i].level - readings[i - 1].level;
            cout << left << setw(8) << readings[i].day
                 << setw(14) << fixed << setprecision(1) << readings[i].level
                 << setw(12) << ch.str() << statusOf(readings[i].level) << "\n";
        }
        double sum = 0, mn = 100, mx = 0;
        for (const auto& r : readings) { sum += r.level; mn = min(mn, r.level); mx = max(mx, r.level); }
        line('-', 50);
        cout << fixed << setprecision(2)
             << "Readings: " << readings.size() << "   Average: " << sum / readings.size()
             << "%   Min: " << mn << "%   Max: " << mx << "%\n";
    }

    // ---------------- 2. add / update ----------------
    void addOrUpdateReading() {
        auto day = input::askInt("\nEnter day number (1-100000): ", 1, 100000);
        if (!day) return;
        auto level = input::askDouble("Enter reservoir level (0-100 %): ", 0, 100);
        if (!level) return;

        auto it = find_if(readings.begin(), readings.end(),
                          [&](const Reading& r) { return r.day == *day; });
        if (it != readings.end()) {
            it->level = *level;
            cout << "Existing reading for day " << *day << " updated.\n";
        } else {
            readings.push_back({*day, *level});
            sort(readings.begin(), readings.end(),
                 [](const Reading& a, const Reading& b) { return a.day < b.day; });
            cout << "Reading added.\n";
        }
    }

    // ---------------- 3. delete ----------------
    void deleteReading() {
        if (readings.empty()) { cout << "\nNothing to delete.\n"; return; }
        auto day = input::askInt("\nEnter day number to delete: ", 1, 100000);
        if (!day) return;
        auto it = remove_if(readings.begin(), readings.end(),
                            [&](const Reading& r) { return r.day == *day; });
        if (it == readings.end()) cout << "No reading found for day " << *day << ".\n";
        else { readings.erase(it, readings.end()); cout << "Reading deleted.\n"; }
    }

    // ---------------- 4. graph ----------------
    void showGraph() const {
        if (readings.empty()) { cout << "\nNo readings to plot.\n"; return; }
        Regression reg = fitLine();

        int lastDay  = readings.back().day;
        int firstDay = max(readings.front().day, lastDay - 29);   // at most 30 days of history
        int endDay   = lastDay + 7;
        int cols     = endDay - firstDay + 1;

        // build canvas: rows 100,95,...,0 ; each day takes 2 characters
        const int rows = 21;
        vector<string> canvas(rows, string(cols * 2, ' '));
        auto rowOf = [&](double level) {
            int r = (int)lround((100.0 - clampPct(level)) / 5.0);
            return max(0, min(rows - 1, r));
        };

        if (reg.valid)
            for (int d = firstDay; d <= endDay; ++d)
                canvas[rowOf(reg.at(d))][(d - firstDay) * 2] = '.';
        if (reg.valid)
            for (int d = lastDay + 1; d <= endDay; ++d)
                canvas[rowOf(reg.at(d))][(d - firstDay) * 2] = '+';
        for (const auto& r : readings)
            if (r.day >= firstDay)
                canvas[rowOf(r.level)][(r.day - firstDay) * 2] = 'o';

        cout << "\n========== RESERVOIR LEVEL GRAPH ==========\n";
        cout << "Level (%)\n";
        for (int i = 0; i < rows; ++i) {
            int level = 100 - i * 5;
            if (level % 10 == 0) cout << setw(3) << level << " |";
            else                 cout << "    |";
            cout << canvas[i] << "\n";
        }
        cout << "    +" << string(cols * 2, '-') << "\n     ";
        for (int d = firstDay; d <= endDay; ++d) {
            if ((d - firstDay) % 5 == 0) {
                string s = to_string(d);
                cout << s << string(max<size_t>(1, 10 - s.size()), ' ');
            }
        }
        cout << "\n     Day number\n\n";
        cout << "  o = recorded reading     . = trend line     + = forecast (next 7 days)\n";
        if (!reg.valid) cout << "  (Add at least two readings to see the trend line.)\n";
    }

    // ---------------- 5. regression forecast ----------------
    void showTrendPrediction() const {
        Regression reg = fitLine();
        if (!reg.valid) { cout << "\nAt least two readings on different days are needed.\n"; return; }
        int lastDay = readings.back().day;

        cout << "\n========== NEXT 7 DAYS - TREND FORECAST ==========\n";
        cout << fixed << setprecision(3)
             << "Trend line : level = " << reg.slope << " x day + " << reg.intercept << "\n"
             << setprecision(1)
             << "Daily change: " << showpos << reg.slope << noshowpos << " % per day"
             << "     Fit quality R^2 = " << setprecision(3) << reg.r2 << "\n";
        cout << "(Estimates only - they continue the existing trend.)\n\n";

        cout << left << setw(10) << "Day" << setw(18) << "Predicted level" << "Status\n";
        line('-', 52);
        for (int i = 1; i <= 7; ++i) {
            double p = clampPct(reg.at(lastDay + i));
            ostringstream pv;
            pv << fixed << setprecision(2) << p << "%";
            cout << left << setw(10) << lastDay + i
                 << setw(18) << pv.str()
                 << statusOf(p) << "\n";
        }
        line('-', 52);

        cout << setprecision(1);
        if (reg.slope < -1e-9) {
            double dLow  = (LOW_LEVEL - reg.intercept) / reg.slope - lastDay;
            double dZero = (0 - reg.intercept) / reg.slope - lastDay;
            if (reg.at(lastDay) > LOW_LEVEL)
                cout << "At this rate the level reaches the LOW mark (" << LOW_LEVEL
                     << "%) in about " << dLow << " days and empties in about " << dZero << " days.\n";
            else
                cout << "Level is already at or below the LOW mark and may empty in about "
                     << max(0.0, dZero) << " days.\n";
        } else if (reg.slope > 1e-9) {
            double dFull = (100 - reg.intercept) / reg.slope - lastDay;
            cout << "Level is rising; it would reach 100% in about " << max(0.0, dFull) << " days.\n";
        } else {
            cout << "Level is steady.\n";
        }
    }

    // ---------------- 6. physical model (RK4) ----------------
    void showPhysicalModel() {
        if (readings.empty()) { cout << "\nAdd at least one reading first.\n"; return; }

        cout << "\n========== PHYSICAL MODEL - RK4 WATER BALANCE ==========\n";
        cout << "  dV/dt = Qin(t) - demand - k*V\n";
        cout << fixed << setprecision(1)
             << "  Capacity " << physics.capacityMm3 << " million m3, release "
             << physics.demand << " m3/s, loss k = " << scientific << setprecision(1)
             << physics.lossCoeff << " /s\n";
        cout << fixed << setprecision(1) << "  Inflow forecast (m3/s), day 0..7: ";
        for (double q : physics.inflowValue) cout << q << " ";
        cout << "\n";

        auto yn = input::askLine("\nEnter your own inflow forecast? (y/n): ");
        if (!yn) return;
        if (!yn->empty() && (*yn)[0] == 'y') {
            for (size_t i = 0; i < physics.inflowValue.size(); ++i) {
                auto q = input::askDouble("  Inflow for day " + to_string(i) + " (m3/s): ", 0, 100000);
                if (!q) return;
                physics.inflowValue[i] = *q;
            }
        }

        const double cap   = physics.capacityMm3 * 1e6;
        const double dt    = 0.25;                        // 6-hour steps
        const int    steps = (int)lround(7 / dt);
        int lastDay = readings.back().day;
        double V0 = readings.back().level / 100.0 * cap;

        numeric::OdeFunc f = [&](double t, double V) {
            double qin = numeric::lerp(t, physics.inflowDays, physics.inflowValue);   // interpolation
            return qin - physics.demand - physics.lossCoeff * V;
        };

        double Vrk = V0, Veu = V0, t = 0;
        cout << "\n" << left << setw(8) << "Day" << setw(14) << "Inflow m3/s"
             << setw(15) << "Volume (Mm3)" << setw(12) << "RK4 level" << setw(12) << "Euler lvl"
             << "Status\n";
        line('-', 74);
        for (int s = 0; s <= steps; ++s) {
            if (s % 4 == 0) {   // print once per day
                double lvl = clampPct(Vrk / cap * 100.0);
                cout << left << setw(8) << lastDay + s / 4
                     << setw(14) << fixed << setprecision(1) << numeric::lerp(t, physics.inflowDays, physics.inflowValue)
                     << setw(15) << setprecision(2) << Vrk / 1e6
                     << setw(12) << setprecision(2) << lvl
                     << setw(12) << clampPct(Veu / cap * 100.0)
                     << statusOf(lvl) << "\n";
            }
            if (s == steps) break;
            Vrk = max(0.0, min(cap, numeric::rk4Step(f, t, Vrk, dt)));   // cannot go below empty / above full
            Veu = max(0.0, min(cap, numeric::eulerStep(f, t, Veu, dt)));
            t += dt;
        }
        line('-', 74);

        // trapezoidal rule: total river inflow over the 7 days
        double inflowVol = numeric::trapezoid(
            [&](double x) { return numeric::lerp(x, physics.inflowDays, physics.inflowValue); },
            0, 7, 700) * SECONDS_PER_DAY;
        double outVol = (physics.demand * 7 * SECONDS_PER_DAY);
        cout << fixed << setprecision(2)
             << "Total inflow  (trapezoidal rule): " << inflowVol / 1e6 << " million m3\n"
             << "Total release (demand x 7 days) : " << outVol / 1e6 << " million m3\n"
             << "Predicted level after 7 days    : " << clampPct(Vrk / cap * 100.0) << " %  ("
             << statusOf(clampPct(Vrk / cap * 100.0)) << ")\n";
    }

    // ---------------- 7. error analysis ----------------
    void showErrorAnalysis() const {
        // test problem with known exact answer:  dV/dt = Q - k V
        //   V(t) = Q/k + (V0 - Q/k) e^(-k t)
        const double Q = 60.0, k = 1.5e-6, V0 = 20e6, T = 8;   // T in days (divisible by every dt below)
        double Vexact = Q / k + (V0 - Q / k) * exp(-k * T * SECONDS_PER_DAY);
        numeric::OdeFunc f = [&](double, double V) { return Q - k * V; };

        cout << "\n========== ERROR ANALYSIS ==========\n";
        cout << "Test: dV/dt = Q - kV with Q = 60 m3/s, k = 1.5e-6 /s, V0 = 20 Mm3, T = 8 days\n";
        cout << "Exact volume after 8 days = " << fixed << setprecision(4) << Vexact / 1e6 << " million m3\n\n";
        cout << left << setw(10) << "dt (day)" << setw(20) << "Euler error (%)" << "RK4 error (%)\n";
        line('-', 50);
        for (double dt : {2.0, 1.0, 0.5, 0.25, 0.125}) {
            int n = (int)lround(T / dt);
            double Ve = V0, Vr = V0, t = 0;
            for (int i = 0; i < n; ++i) {
                Ve = numeric::eulerStep(f, t, Ve, dt);
                Vr = numeric::rk4Step(f, t, Vr, dt);
                t += dt;
            }
            cout << left << setw(10) << fixed << setprecision(3) << dt
                 << setw(20) << scientific << setprecision(3) << fabs(Ve - Vexact) / Vexact * 100
                 << fabs(Vr - Vexact) / Vexact * 100 << "\n";
        }
        cout << "\n-> Halving dt halves the Euler error (1st order) but cuts RK4 error ~16x (4th order).\n";

        cout << "\nTrapezoidal rule check: integral of sin(x) on [0, pi]  (exact = 2)\n";
        cout << left << setw(10) << "strips" << setw(18) << "value" << "error\n";
        line('-', 44);
        for (int n : {4, 8, 16, 32, 64}) {
            double v = numeric::trapezoid([](double x) { return sin(x); }, 0, PI, n);
            cout << left << setw(10) << n << setw(18) << fixed << setprecision(8) << v
                 << scientific << setprecision(2) << fabs(v - 2) << "\n";
        }
        cout << "\n-> Doubling the strips cuts the trapezoid error ~4x (2nd order).\n";
    }

    // ---------------- 8 / 9. file I/O ----------------
    void saveToFile() const {
        ofstream out(DATA_FILE);
        if (!out) { cout << "\nCould not write " << DATA_FILE << ".\n"; return; }
        out << "day,level\n";
        for (const auto& r : readings) out << r.day << "," << r.level << "\n";
        cout << "\nSaved " << readings.size() << " readings to " << DATA_FILE << ".\n";
    }

    void loadFromFile() {
        ifstream in(DATA_FILE);
        if (!in) { cout << "\nFile " << DATA_FILE << " not found. Save some data first (option 8).\n"; return; }
        vector<Reading> loaded;
        string row;
        getline(in, row);   // header
        while (getline(in, row)) {
            size_t comma = row.find(',');
            if (comma == string::npos) continue;
            try {
                int d = stoi(row.substr(0, comma));
                double l = stod(row.substr(comma + 1));
                if (d > 0 && l >= 0 && l <= 100) loaded.push_back({d, l});
            } catch (...) {}
        }
        if (loaded.empty()) { cout << "\nNo valid readings found in the file.\n"; return; }
        sort(loaded.begin(), loaded.end(), [](const Reading& a, const Reading& b) { return a.day < b.day; });
        readings = loaded;
        cout << "\nLoaded " << readings.size() << " readings from " << DATA_FILE << ".\n";
    }
};

int main() {
    ReservoirSystem system;
    system.run();
    return 0;
}
