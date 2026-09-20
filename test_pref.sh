#!/bin/bash

# Setup base folders
CHAMPSIM_DIR="$HOME/ChampSim"
cd "$CHAMPSIM_DIR" || exit 1

mkdir -p results
mkdir -p build_logs

CONFIG_FILE="champsim_config.json"
TRACE_FILE="traces/600.perlbench_s-210B.champsimtrace.xz"
WARMUP=100000
SIM=1000000

# Verify trace exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "[ERROR] Trace file not found at $TRACE_FILE"
    exit 1
fi

# Full roster of unique L2 candidate modules deployed earlier
MODULES=(
    "no"
    "next_line"
    "ip_stride"
    "stride_pythia"
    "streamer"
    "stream_dpc2"
    "bop"
    "mlop"
    "sms"
    "ampm"
    "ampm_lite"
    "bingo"
    "bingo_plus"
    "dspatch"
    "spp_dev"
    "spp_dev2"
    "sppam"
    "ipcp_l1"
    "ipcp_l2"
    "sberti"
    "gberti"
    "bertigo"
    "berti_plus"
    "berti_vip"
    "edp"
    "infollc"
    "anelin"
    "emender_l1d"
    "emender_l2"
    "emender_l3"
    "umama"
    "pythia"
    "ppf_dev"
    "scooby"
    "sandbox"
    "pref_power7"
)

echo "============================================================"
echo "STARTING L2C PREFETCHER TEST HARNESS"
echo "============================================================"

# Prepare scoreboard log
SUMMARY_LOG="results/l2c_test_summary.txt"
echo "L2C PREFETCHER TEST REPORT ($(date))" > "$SUMMARY_LOG"
echo "------------------------------------------------------------" >> "$SUMMARY_LOG"
printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "MODULE" "CONFIG/BUILD" "SIMULATION" "REQUESTED" "USEFUL" >> "$SUMMARY_LOG"
echo "------------------------------------------------------------" >> "$SUMMARY_LOG"

for MOD in "${MODULES[@]}"; do
    echo ""
    echo ">>> TESTING MODULE: $MOD"
    
    # 1. Check if folder exists
    if [ ! -d "prefetcher/$MOD" ]; then
        echo "[-] Folder prefetcher/$MOD does not exist. Skipping."
        printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "NO_DIR" "SKIPPED" "-" "-" >> "$SUMMARY_LOG"
        continue
    fi

    # 2. Modify config: Set L1D to 'no', L2C to module name using python
    python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    data = json.load(f)
data['L1D']['prefetcher'] = 'no'
data['L2C']['prefetcher'] = '$MOD'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"

    # 3. Configure ChampSim
    ./config.sh "$CONFIG_FILE" > "build_logs/${MOD}_config.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "[-] ./config.sh FAILED for $MOD. See build_logs/${MOD}_config.log"
        printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "CONFIG_FAIL" "SKIPPED" "-" "-" >> "$SUMMARY_LOG"
        continue
    fi

    # 4. Compile ChampSim
    make -j$(nproc) > "build_logs/${MOD}_make.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "[-] make FAILED for $MOD (Compilation/Link Error). See build_logs/${MOD}_make.log"
        printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "BUILD_FAIL" "SKIPPED" "-" "-" >> "$SUMMARY_LOG"
        continue
    fi
    echo "[+] Build SUCCESS for $MOD."

    # 5. Run simulation
    OUT_FILE="results/perlbench_210B_L2C_${MOD}_physical_1core_${WARMUP}warm_${SIM}sim.txt"
    ./bin/champsim --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TRACE_FILE" > "$OUT_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[-] Simulation CRASHED/FAILED for $MOD."
        printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "BUILD_OK" "CRASHED" "-" "-" >> "$SUMMARY_LOG"
        continue
    fi

    # 6. Check if prefetcher actually operated
    PREF_LINE=$(grep "cpu0->cpu0_L2C PREFETCH REQUESTED:" "$OUT_FILE" | tail -n 1)
    if [ -z "$PREF_LINE" ]; then
        echo "[-] Simulation completed but output missing L2C prefetch statistics."
        printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "BUILD_OK" "NO_STATS" "-" "-" >> "$SUMMARY_LOG"
    else
        REQ=$(echo "$PREF_LINE" | awk '{print $4}')
        USEFUL=$(echo "$PREF_LINE" | awk '{print $8}')
        
        if [ "$REQ" -eq 0 ] && [ "$MOD" != "no" ]; then
            echo "[!] Simulation PASSED, but INACTIVE (0 prefetches issued)."
            printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "BUILD_OK" "INACTIVE(0)" "$REQ" "$USEFUL" >> "$SUMMARY_LOG"
        else
            echo "[+] Simulation PASSED and ACTIVE. Requested: $REQ | Useful: $USEFUL"
            printf "%-18s | %-12s | %-12s | %-10s | %-10s\n" "$MOD" "BUILD_OK" "SUCCESS" "$REQ" "$USEFUL" >> "$SUMMARY_LOG"
        fi
    fi

done

echo ""
echo "============================================================"
echo "TEST SUITE FINISHED. SUMMARY TABLE:"
echo "============================================================"
cat "$SUMMARY_LOG"
