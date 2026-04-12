# Huffman Cluster Slot Replacement Strategy Comparison

Issue: #38

## Summary

All 5 strategies produce file sizes within 0.03% of each other.
The issue's "~1-2%" estimate is incorrect — real impact is ~100x smaller.

**Recommendation**: Keep RoundRobin (C++ parity). No strategy consistently
wins across all corpora and quality levels.

## Strategies Tested

| Strategy | Description |
|----------|-------------|
| RoundRobin | (default) Cycle through slots 0-3 — matches C++ jpegli |
| SmallestCount | Evict slot with fewest total symbols |
| LowestEvictionCost | Evict slot cheapest to re-merge into another |
| OldestSlot | Evict least recently assigned/used slot |
| HighestSlotCost | Evict slot with highest current encoding cost |

## Results: 4:2:0 Progressive (default jpegli scan script)

```
Corpus   Q        RoundRobin  SmallestCnt  LowestEvict   OldestSlot  HighestCost  best
----------------------------------------------------------------------------------------------------
sc       50          1268270     +0.0142%     +0.0142%     +0.0000%     +0.0027%  RoundRobin
sc       75          1760041     +0.0088%     +0.0088%     +0.0000%     +0.0000%  RoundRobin
sc       85          2147182     +0.0135%     +0.0135%     +0.0000%     +0.0021%  RoundRobin
sc       95          3161592     +0.0035%     +0.0035%     +0.0000%     +0.0033%  RoundRobin
cid      50          4689198     +0.0050%     +0.0050%     +0.0000%     +0.0121%  RoundRobin
cid      75          7022446     +0.0114%     +0.0113%     +0.0000%     +0.0119%  RoundRobin
cid      85          9218908     +0.0148%     +0.0146%     +0.0000%     +0.0175%  RoundRobin
cid      95         16742021     +0.0133%     +0.0134%     +0.0000%     +0.0159%  RoundRobin
clic     50          5165283     +0.0156%     +0.0154%     +0.0000%     +0.0045%  RoundRobin
clic     75          7909805     +0.0062%     +0.0062%     +0.0000%     +0.0063%  RoundRobin
clic     85         10840645     +0.0073%     +0.0073%     +0.0000%     +0.0055%  RoundRobin
clic     95         21856373     +0.0023%     +0.0023%     +0.0000%     -0.0006%  HighestCost
```

RoundRobin wins 11/12 cases. OldestSlot ties exactly. HighestCost wins once
(clic Q95) by 13 bytes out of 21.8MB.

## Results: 4:2:0 with optimize_scans (more complex scan scripts)

```
Corpus   Q        RoundRobin  SmallestCnt  LowestEvict   OldestSlot  HighestCost  best
----------------------------------------------------------------------------------------------------
sc       50          1247650     +0.0042%     +0.0042%     +0.0000%     +0.0000%  RoundRobin
sc       75          1742439     +0.0021%     +0.0021%     +0.0000%     +0.0000%  RoundRobin
sc       85          2133271     +0.0016%     +0.0016%     +0.0000%     -0.0015%  HighestCost
sc       95          3132183     +0.0032%     +0.0032%     +0.0000%     +0.0013%  RoundRobin
cid      50          4594992     +0.0108%     +0.0108%     +0.0000%     +0.0033%  RoundRobin
cid      75          6950610     +0.0151%     +0.0146%     +0.0000%     +0.0018%  RoundRobin
cid      85          9168860     +0.0131%     +0.0130%     -0.0002%     +0.0080%  OldestSlot
cid      95         16690329     +0.0083%     +0.0085%     -0.0002%     +0.0154%  OldestSlot
clic     50          5083519     +0.0021%     +0.0021%     +0.0000%     +0.0000%  RoundRobin
clic     75          7845307     +0.0006%     +0.0006%     +0.0000%     +0.0008%  RoundRobin
clic     85         10788421     +0.0040%     +0.0029%     +0.0000%     +0.0033%  RoundRobin
clic     95         21744290     +0.0010%     +0.0010%     +0.0000%     -0.0002%  HighestCost
```

OldestSlot wins 2/12 (cid Q85/Q95) by 0.0002%. HighestCost wins 2/12 by
similar margin. RoundRobin wins 8/12.

## Results: 4:4:4 Progressive (more AC contexts per scan)

```
Corpus   Q        RoundRobin  SmallestCnt  LowestEvict   OldestSlot  HighestCost  best
----------------------------------------------------------------------------------------------------
sc       50          1504522     +0.0055%     +0.0055%     -0.0001%     +0.0011%  OldestSlot
sc       75          2023889     +0.0043%     +0.0043%     +0.0016%     +0.0016%  RoundRobin
sc       85          2452086     +0.0063%     +0.0063%     -0.0015%     -0.0009%  OldestSlot
sc       95          3552946     +0.0051%     +0.0051%     +0.0000%     +0.0000%  RoundRobin
cid      50          5545400     +0.0091%     +0.0076%     -0.0000%     +0.0181%  OldestSlot
cid      75          8171168     +0.0211%     +0.0211%     -0.0003%     +0.0183%  OldestSlot
cid      85         10743531     +0.0277%     +0.0277%     +0.0000%     +0.0138%  RoundRobin
cid      95         19242252     +0.0193%     +0.0202%     -0.0001%     +0.0120%  OldestSlot
clic     50          6252348     +0.0148%     +0.0148%     -0.0012%     -0.0006%  OldestSlot
clic     75          9464849     +0.0080%     +0.0080%     +0.0000%     +0.0005%  RoundRobin
clic     85         13003700     +0.0056%     +0.0056%     -0.0002%     -0.0001%  OldestSlot
clic     95         25158268     +0.0024%     +0.0024%     +0.0000%     -0.0003%  HighestCost
```

OldestSlot wins 7/12 by up to 0.0015%. But SmallestCount and
LowestEvictionCost are consistently WORSE (up to +0.028%).

## Why the Impact Is So Small

1. **Few evictions**: 3-component progressive has ~9 AC contexts. After
   filling 4 slots, only 5 evictions occur. Each eviction changes one
   slot assignment.

2. **Tables are rebuilt per-cluster**: Each new cluster gets its own
   optimized Huffman tree. The replacement only affects which slot ID
   it gets, which determines if a new DHT marker must be emitted or
   if an existing one is redefined.

3. **DHT overhead is small**: A DHT marker is ~30-400 bytes depending
   on symbol count. At typical file sizes (100KB-1MB), the cost of an
   extra or suboptimal table is negligible.

4. **Merge decisions dominate**: The algorithm's merge-vs-create decision
   (which is identical across strategies) determines ~99.97% of the
   encoding efficiency. The slot assignment for the remaining creates
   is a rounding error.

## Consistently Worse Strategies

**SmallestCount** and **LowestEvictionCost** are always worse than
RoundRobin, typically by +0.005% to +0.028%. The "evict the smallest"
heuristic backfires because small clusters often represent niche symbol
distributions (e.g., chroma refinement scans) that are cheap to keep
in their own table.

## OldestSlot = RoundRobin for 4:2:0

OldestSlot produces byte-identical output to RoundRobin for 4:2:0
progressive. This happens because contexts are processed sequentially
and the "oldest unreplaced slot" is always the next slot in the
round-robin cycle. They diverge slightly for 4:4:4 where merge
patterns differ.
