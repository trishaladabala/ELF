# Phase 0L Follow-Up Audit

**Date:** 2026-07-21 14:34

## 1. Updated Degeneracy Audit
- Sequences checked: 32 (L=128)
- Templated/n-gram repetition rate: **43.8%** (95% CI: [28.2%, 60.7%])
- Non-degenerate PPL at L=128: 144.5
- Raw PPL (all L=128): 132.4

## 2. Context-Length PPL Check
- Sequences checked: 8 (L=1024)
- Non-degenerate PPL at L=1024: **28.5**
- Raw PPL (all L=1024): **26.3**
- Paper Reference: 24.1

## Verdict
**PASS: PPL at L=1024 confirms context length was the primary gap.**

## Appendix: Sample L=1024 Texts

### Sample 1 [PPL=33.8]
```
Morrison pointed out that in the new world, renewable energy producers are generate more turnover in the years than expected thanks to increased costs from renewable energy and increasing demand. But Morrison pointed out that increased benefits from renewable energy are outweighed by increasing demand for renewables. Morrison told Business Inside that that the world's largest economies are struggling to meet energy demand, especially in the regions areas of China and South Asia. She also told me that at a time when there is be increase in energy prices, there's going to be increasing demand for renewable energy throughout the world. "This is a very big problem because we are increasing demand for renewable energy every day," she said. "If we can't deal with sources, if we can't deal with o...
```

### Sample 2 [PPL=23.6]
```
Morrison pointed out that experts predicted North Korea’s nuclear capacity would be between 2.5 and 0.5 percent higher due to increased interest in China’s nuclear program, while Morrison pointed attention to the fact that both China and the U.S. have close relations with North Korea. Morrison also pointed out that both countries have expressed very little interest in developing a nuclear test, most recently on North Korea at the UN Security Council. She pointed out that North Korea is for projecting an ambitious nuclear test in the Middle East and Pacific Sea that will increase the opportunities for nuclear forces in the region. “That is a pretty big issue. We’ve got to fix it all along,” she said. “Now we have to work with the Korean leaders, but they’re not going to fix it. That’s a big...
```

### Sample 3 [DEG]
```
Die Laut darauf Wir Deutschen wird das wird wird Wasser Wasser Wasser Kosten wird wird zu zu zu zu zu Euro werden wird. wird wird für die Wasser aus. Wasser Wasser wird. wird wird die Wasser wird wird wird Wasser werden wird wird wird ist aus aus aus die Wasser aus Wasser Wasser Wasser. Laut Wasser Laut Laut „ „ Wasser ist „ nicht nicht nicht zu zu zu Wasser ist wird Wasser Wasser aus Wasser Wasser Wasser Wasser Wasser. Wasser Wasser Wasser Wasser,. Da wird wird wird wird wird wird wird wird zu wird zu zu Wasser Wasser zu Wasser wird wird zu wird zu zu zu Wasser Wasser zu. Aus. „Das ist ist ista große Sache. „. nicht gut zu geht zu geht“,, He sagte. " „ Es sind nicht nicht nicht nicht nicht nicht nicht nicht ist nicht nicht nicht nicht nicht nicht.... Das ist ist gut Sache, damit ist zu zu...
```
