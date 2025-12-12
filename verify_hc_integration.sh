#!/bin/bash
# HC Integration Verification Script
# Run this to verify all HC files are present and correct

echo "=========================================="
echo "HC Integration Verification"
echo "=========================================="
echo ""

ERRORS=0

# Check HC module files
echo "Checking HC module files..."
for file in __init__.py higher_criticism.py null_distribution.py build_null_dist.py; do
    if [ -f "commaqa/hc/$file" ]; then
        echo "✓ commaqa/hc/$file"
    else
        echo "✗ Missing: commaqa/hc/$file"
        ERRORS=$((ERRORS+1))
    fi
done
echo ""

# Check HC retrieval participant
echo "Checking HC retrieval participant..."
if [ -f "commaqa/inference/hc_retrieval.py" ]; then
    echo "✓ commaqa/inference/hc_retrieval.py"
else
    echo "✗ Missing: commaqa/inference/hc_retrieval.py"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check config files (18 total)
echo "Checking HC config files..."
CONFIG_COUNT=$(ls base_configs/hc_qa_*.jsonnet 2>/dev/null | wc -l)
if [ "$CONFIG_COUNT" -eq 18 ]; then
    echo "✓ All 18 config files present"
else
    echo "✗ Expected 18 config files, found $CONFIG_COUNT"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check modified files
echo "Checking modified files..."

if grep -q "hc_retrieve_and_select" commaqa/inference/constants.py; then
    echo "✓ constants.py: HC participant registered"
else
    echo "✗ constants.py: HC participant NOT registered"
    ERRORS=$((ERRORS+1))
fi

if grep -q '"hc_qa"' run.py; then
    echo "✓ run.py: hc_qa in instantiation_schemes"
else
    echo "✗ run.py: hc_qa NOT in instantiation_schemes"
    ERRORS=$((ERRORS+1))
fi

if grep -q 'hc_qa' runner.py; then
    echo "✓ runner.py: hc_qa in system choices"
else
    echo "✗ runner.py: hc_qa NOT in system choices"
    ERRORS=$((ERRORS+1))
fi

if grep -q 'hc_qa' run_retrieval_dev.sh; then
    echo "✓ run_retrieval_dev.sh: hc_qa in valid_systems"
else
    echo "✗ run_retrieval_dev.sh: hc_qa NOT in valid_systems"
    ERRORS=$((ERRORS+1))
fi

if grep -q 'hc_qa' run_retrieval_test.sh; then
    echo "✓ run_retrieval_test.sh: hc_qa in valid_systems"
else
    echo "✗ run_retrieval_test.sh: hc_qa NOT in valid_systems"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED"
    echo "=========================================="
    echo ""
    echo "Integration is complete and correct!"
    echo ""
    echo "Next steps:"
    echo "1. Build null distribution:"
    echo "   python -m commaqa.hc.build_null_dist hotpotqa dev_500"
    echo ""
    echo "2. Run HC on dev set:"
    echo "   bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010"
    exit 0
else
    echo "✗ FOUND $ERRORS ERROR(S)"
    echo "=========================================="
    echo ""
    echo "Please check the missing files above."
    exit 1
fi
