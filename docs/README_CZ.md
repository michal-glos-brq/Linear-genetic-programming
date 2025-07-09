
### Lineární Genetické Programování - Python Modul

Tento modul poskytuje implementaci Lineárního Genetického Algoritmu (LGA) pro klasifikaci obrázků. Umožňuje uživatelům konfigurovat a spouštět experimenty s různými konfiguracemi.

#### Requires Python3.8+
##### Anaconda is the preffered way to install the application
`pip install -r requirements.txt` 

CLI argumenty:

- Argumenty související s datasetem
  - `-d`, `--dataset`:  Vyberte dataset z torchvision, např. MNIST, CIFAR10, atd. (výchozí: MNIST)
  - `--data-dir`: Zadejte adresář pro uložení datasetu
  - `--resize`: Změňte velikost obrázků na zadaný čtverec (velikost hran v pixelech)
  - `--split`: Zadejte procento datasetu použitého pro trénink (0-100)
  - `-n`, `--normalize`: Zadejte interval pro normalizaci datasetu (např. 0 1 nebo -1 1)
  - `--test`: Vytvořte vlastní testovací dataset zadáním počtu tříd a datových záznamů (např. 10 1000)

 - Argumenty související s tokem dat
   - `-l`, `--load`:  Načtěte předtrénovaný program ze zadané cesty
   - `-md`, `--model-dir`:  Zadejte adresář pro uložení nejlepších programů ve formátu: {dataset}{resize}{fitness}_{timestamp}.p
   - `-log`, `--logging-dir`: Zadejte adresář pro záznamy

 - Hyperparametry algoritmu
   - `-p`, `--population`: Zadejte velikost populace (výchozí: 42)
   - `-g`, `--gens`: Zadejte počet generací, po které se LGA bude vyvíjet (výchozí: 60)
   - `--runs`: Zadejte počet opakování algoritmu (výchozí: 10)
   - `-mini`, `--min-instructions`: Zadejte minimální počet instrukcí pro vyvíjené programy
   - `-maxi`, `--max-instructions`: Zadejte maximální počet instrukcí pro vyvíjené programy
   - `-f`, `--fitness`: Vyberte fitness funkci (viz níže pro dostupné možnosti)
   -  `-pg`, `--p-grow`: Zadejte šanci (v %) na postupné zvýšení počtu instrukcí programu (výchozí: 25)
   -  `-ps`, `--p-shrink`: Zadejte šanci (v %) na postupné snížení počtu instrukcí programu (výchozí: 25)
   -  `-pm`, `--p-mutate`: Zadejte šanci (v %) na mutaci programu
   -  `-pc`, `--p-cross`: Zadejte šanci (v %) na křížení při vytváření nových potomků
   -  `-pa`, `--p-area`:  Zadejte pravděpodobnost (v %) použití instrukce s tenzorovým výřezem místo skalárních hodnot
   -  `--mutate-regs`: Zadejte max. počet hodnot registrů pro mutaci (výchozí: 1)
   -  `--mutate-inst`: Zadejte max. počet instrukcí k mutaci (výchozí: 1)
   -  `--elite`: Elita, která má být uchována po výběru
   -  `--elite-equal`: Vzorek elitních jedinců pro křížení a výběr rovnoměrně, bez ohledu na jejich fitness
   -  `-r`, `--regs`: Zadejte tvar pracovních registrů jako n-tici celých čísel (výchozí: (42,))
   -  `-b`, `--binary`: Vyberte binární operace použité v lineárním programu
   -  `-u`, `--unary`: Vyberte unární operace použité v lineárním programu
   -  `-a`, `--area`: Vyberte plošné operace použité v lineárním programu (redukující tensor na skalární hodnotu)


- ostatní parametry
   -  `--debug`: Povolte záznamy úrovně DEBUG 
