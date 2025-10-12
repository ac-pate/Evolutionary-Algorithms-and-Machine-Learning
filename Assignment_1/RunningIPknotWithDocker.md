# Running IPknot with Docker

This guide explains how to build and run [IPknot](https://github.com/satoken/ipknot) using Docker.  
It covers cloning the repository, fixing the Dockerfile, and running the program with example input.

---

## 1. Clone the repository

```powershell
git clone https://github.com/satoken/ipknot.git
cd ipknot
```

## 2. Install Docker Desktop for Windows

Download and install from: https://docs.docker.com/desktop/setup/install/windows-install/
or https://docs.docker.com/desktop/setup/install/mac-install/

## 3. Try building with the default Dockerfile

```powershell
docker build . -t ipknot
```
Then test run:

```powershell
docker run -it --rm -v ${PWD}:/work -w /work ipknot examples/RF00005.fa
```

If this works, you’re done. \
If it fails with build errors, continue to Step 4.


## 4. Fix the Dockerfile

Replace the existing `Dockerfile` with this content:

```dockerfile
FROM satoken/viennarna:latest AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git pkg-config zlib1g-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ERGO-Code/HiGHS /tmp/HiGHS && \
    cmake -S /tmp/HiGHS -B /tmp/HiGHS/build -G Ninja \
          -DFAST_BUILD=ON -DBUILD_SHARED_LIBS=ON && \
    cmake --build /tmp/HiGHS/build --parallel && \
    cmake --install /tmp/HiGHS/build

WORKDIR /src
COPY . /src
RUN cmake -S . -B build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_HIGHS=ON && \
    cmake --build build --parallel && \
    cmake --install build --strip

FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 zlib1g && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/ /usr/local/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENTRYPOINT ["ipknot"]

```

## 5. Build again

```powershell
docker build . -t ipknot
```

## 6. Run with example input

```powershell
docker run -it --rm -v ${PWD}:/work -w /work ipknot examples/RF00005.fa
```

If you see the result like the following, you are good to go. 

```
PS C:\Users\kaiyu\Desktop\COEN432\ipknot> docker run -it --rm -v ${PWD}:/work -w /work ipknot examples/RF00005.fa
>J01390-1/6861-6932
CAGGUUAGAGCCAGGUGGUUAGGCGUCUUGUUUGGGUCAAGAAAUUGUUAUGUUCGAAUCAUAAUAACCUGA
((((((...(((.........))).(((((.......)))))....((((((.......)))))))))))).
>J05395-1/2325-2252
GGUUUCGUGGUCUAGUCGGUUAUGGCAUCUGCUUAACACGCAGAACGUCCCCAGUUCGAUCCUGGGCGAAAUCG
(((((((........[[[.[[......(((((..]]...))))).....(((((..]]]..)))))))))))).
>K00228-1/1-82
GGUUGUUUGGCCGAGCGGUCUAAGGCGCCUGAUUCAAGCUCAGGUAUCGUAAGAUGCAAGAGUUCGAAUCUCUUAGCAACCA
(((((((..(((...........)))..[[(((((.(((((]](((((....)))))[[))))).)))))..]]))))))).
>AC009395-7/99012-98941
GGCUCAAUGGUCUAGGGGUAUGAUUCUCGCUUUGGGUGCGAGAGGUCCCGGGUUCAAAUCCCGGUUGAGCCC
((((((((.......(([[.[[[(((((((.......)))))))..))((((.]]].]])))))))))))).
>J04815-1/3159-3231
AGAGCUUGCUCCCAAAGCUUGGGUGUCUAGCUGAUAAUUAGACUAUCAAGGGUUAAAUUCCCUUCAAGCUCUA
((((((((..((((.....)))).((((((..[[[[.))))))]]]](((((.......))))))))))))).
>M20972-1/1-72
AGGGCUAUAGCUCAGCGGUAGAGCGCCUCGUUUACACCGAGAAUGUCUACGGUUCAAAUCCGUAUAGCCCUA
((((((((([[[[.....((]]]]..((((.......))))......))(((.......)))))))))))).
>M68929-1/151018-150946
CGCGGGAUAGAGUAAUUGGUAACUCGUCAGGCUCAUAAUCUGAAUGUUGUGGGUUCGAAUCCGACUCCCGCCA
.((((((..[[[[...(((..]]]].[[[((((((((((.]]]..)))))))))).....)))..))))))..
>X00360-1/1-73
CCGACCUUAGCUCAGUUGGUAGAGCGGAGGACUGUAGAUCCUUAGGUCACUGGUUCGAAUCCGGUAGGUCGGA
(((((((..((((........))))[[[[[[[[...((((....))))(((]]]]]...]]])))))))))).
>X12857-1/421-494
GCGGAUGUAGCCAAGUGGAUCAAGGCAGUGGAUUGUGAAUCCACCAUGCGCGGGUUCAAUUCCCGUCAUUCGCC
(((((((........((((.....(((((((((.....))))))..)))[[[[[))))...]]]]]))))))).
>M16863-1/21-94
GGGCUCGUAGCUCAGAGGAUUAGAGCACGCGGCUACGAACCACGGUGUCGGGGGUUCGAAUCCCUCCUCGCCCA
.((.((((((((..................)))))))).))..((((..(((((.......)))))..))))..
```

## 7. Use Docker with python

There are more than one ways to use Docker in parallel with python. Some methods are faster than the others. If you think the method presented here is not fast enough for you or you want to push to the limit, feel free to explore on your own. 

The docker `exec` command runs a new command in a running container. https://docs.docker.com/reference/cli/docker/container/exec/

```powershell
 docker rm ipknot
 docker run -dit --name ipknot_runner -v ${PWD}:/work -w /work --entrypoint bash ipknot -c "sleep infinity"
 docker exec ipknot_runner ipknot examples/RF00005.fa
 ```

You should see the following as results
```powershell
PS C:\Users\kaiyu\Desktop\COEN432\ipknot> docker exec ipknot_runner ipknot examples/RF00005.fa
>J01390-1/6861-6932
CAGGUUAGAGCCAGGUGGUUAGGCGUCUUGUUUGGGUCAAGAAAUUGUUAUGUUCGAAUCAUAAUAACCUGA
((((((...(((.........))).(((((.......)))))....((((((.......)))))))))))).
>J05395-1/2325-2252
GGUUUCGUGGUCUAGUCGGUUAUGGCAUCUGCUUAACACGCAGAACGUCCCCAGUUCGAUCCUGGGCGAAAUCG
(((((((........[[[.[[......(((((..]]...))))).....(((((..]]]..)))))))))))).
>K00228-1/1-82
GGUUGUUUGGCCGAGCGGUCUAAGGCGCCUGAUUCAAGCUCAGGUAUCGUAAGAUGCAAGAGUUCGAAUCUCUUAGCAACCA
(((((((..(((...........)))..[[(((((.(((((]](((((....)))))[[))))).)))))..]]))))))).
>AC009395-7/99012-98941
GGCUCAAUGGUCUAGGGGUAUGAUUCUCGCUUUGGGUGCGAGAGGUCCCGGGUUCAAAUCCCGGUUGAGCCC
((((((((.......(([[.[[[(((((((.......)))))))..))((((.]]].]])))))))))))).
>J04815-1/3159-3231
AGAGCUUGCUCCCAAAGCUUGGGUGUCUAGCUGAUAAUUAGACUAUCAAGGGUUAAAUUCCCUUCAAGCUCUA
((((((((..((((.....)))).((((((..[[[[.))))))]]]](((((.......))))))))))))).
>M20972-1/1-72
AGGGCUAUAGCUCAGCGGUAGAGCGCCUCGUUUACACCGAGAAUGUCUACGGUUCAAAUCCGUAUAGCCCUA
((((((((([[[[.....((]]]]..((((.......))))......))(((.......)))))))))))).
>M68929-1/151018-150946
CGCGGGAUAGAGUAAUUGGUAACUCGUCAGGCUCAUAAUCUGAAUGUUGUGGGUUCGAAUCCGACUCCCGCCA
.((((((..[[[[...(((..]]]].[[[((((((((((.]]]..)))))))))).....)))..))))))..
>X00360-1/1-73
CCGACCUUAGCUCAGUUGGUAGAGCGGAGGACUGUAGAUCCUUAGGUCACUGGUUCGAAUCCGGUAGGUCGGA
(((((((..((((........))))[[[[[[[[...((((....))))(((]]]]]...]]])))))))))).
>X12857-1/421-494
GCGGAUGUAGCCAAGUGGAUCAAGGCAGUGGAUUGUGAAUCCACCAUGCGCGGGUUCAAUUCCCGUCAUUCGCC
(((((((........((((.....(((((((((.....))))))..)))[[[[[))))...]]]]]))))))).
>M16863-1/21-94
GGGCUCGUAGCUCAGAGGAUUAGAGCACGCGGCUACGAACCACGGUGUCGGGGGUUCGAAUCCCUCCUCGCCCA
.((.((((((((..................)))))))).))..((((..(((((.......)))))..))))..
```

To use this command with `python`, you can check the python code below as a reference. 


```python
# test_parallel_output.py
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 8
files = ["/work/examples/RF00005.fa"] * N   # run same input 8 times

def run_one(f):
    cmd = ["docker", "exec", "ipknot_runner", "ipknot", f]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout

results = []
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = [pool.submit(run_one, f) for f in files]
    for fut in as_completed(futs):
        res = fut.result()
        results.append(res)
        # print each result immediately
        print("=== One job finished ===")
        print(res.strip(), "\n")

print(f"Ran {len(results)} jobs total")
```



If the current performance does not meet your requirements, you are expected to explore optimization strategies independently. Please note that computational speed is a critical factor in the effectiveness of evolutionary algorithms.


















