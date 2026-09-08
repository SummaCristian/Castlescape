# Glossario

Definizioni brevi dei termini di Computer Graphics usati in `OVERVIEW.md`,
`notes.md` e nei commenti del codice. Ordinato per argomento. Per ogni voce:
cos'è, e dove/perché compare nel progetto.

---

## GPU e shader

**GPU** — processore che esegue lo stesso piccolo programma su moltissimi dati
in parallelo. Non decide *cosa* disegnare (quello è la CPU, `main.cpp`): riceve
vertici, matrici e texture e li trasforma in pixel.

**Vertex shader** — programma che gira una volta per vertice. Decide dove
finisce il vertice sullo schermo (`gl_Position`) e passa dati al fragment
shader. Nel progetto: `PosNormUV.vert`, `Flame.vert`, ecc.

**Fragment shader** — programma che gira una volta per pixel coperto da un
triangolo. Calcola il colore di quel pixel. Cuore del progetto:
`CookTorrance.frag`.

**Primitiva** — l'unità che la GPU rasterizza: triangolo, linea o punto. Le
linee di debug (`DebugLines`) usano `LINE_LIST`, tutto il resto triangoli.

**Rasterizzazione** — passo fisso della pipeline che trasforma un triangolo
(3 vertici) nell'insieme di pixel che copre, interpolando i dati dei vertici
su ognuno.

**Interpolazione (varying / `in`/`out`)** — i valori che il vertex shader
scrive in `out` non arrivano identici al fragment shader: la GPU li mescola tra
i tre vertici del triangolo. È così che tre normali ai vertici diventano una
superficie liscia sfumata.

**dFdx / dFdy** — derivate di una variabile tra pixel adiacenti. Usate in
`CookTorrance.frag` per ricavare la normale della faccia dal gradiente della
posizione mondo (`flatNormals`).

**Vertex pulling** — trucco per disegnare senza vertex buffer: gli indici dei
vertici stanno in un uniform buffer come array e lo shader si cerca il proprio
dato con `gl_VertexIndex`. Usato da `DebugLines`, perché una classe nostra non
può accedere a `createBuffer()` di `Starter.hpp`.

---

## Vulkan: come i dati arrivano allo shader

**Uniform Buffer Object (UBO)** — blocco di memoria uniforme (uguale per tutti
i vertici/pixel di una draw) che lo shader legge come una `struct`. Camera,
luci, matrici e materiali passano tutti da UBO.

**Push constant** — piccolo blocco di dati (max ~128 byte) infilato
direttamente nel command buffer. Veloce ma **congelato al momento della
registrazione**: va bene solo per valori che non cambiano mai (es. l'indice
faccia in `ShadowCube.vert`), non per qualcosa che si muove ogni frame.

**Descriptor set** — l'oggetto che dice allo shader *dove* stanno gli UBO e le
texture che gli servono. Raggruppati per frequenza di aggiornamento: set 0 =
una volta a frame (`gubo`), set 1 = una volta a oggetto (`ubo`), set 2 = una
volta per luce che proietta ombra.

**Descriptor set layout** — lo schema di un descriptor set: quanti binding,
di che tipo, per quale stage. La pipeline lo vuole per essere costruita.

**Descriptor pool** — riserva da cui vengono allocati i descriptor set.
Dimensionata *prima* di essere creata, quindi il numero massimo di fiamme,
ombre, ecc. va deciso in anticipo (`DPSZs.uniformBlocksInPool` / `setsInPool`).

**Binding** — lo slot numerato dentro un set (`layout(binding = 0, set = 1)`).

**std140** — le regole di impaginazione della memoria di un blocco uniforme.
Un `vec3` è allineato a 16 byte, un `mat3` diventa tre colonne allineate
separatamente, ecc. Vanno rispettate identiche in C++ e in GLSL o i due
smettono di descrivere gli stessi byte, senza errore. Per questo `nMat` è un
`mat4` e non un `mat3`, e `alignas(16)` compare sui `vec3` nelle struct C++.

**Allineamento / padding** — spazio vuoto che std140 inserisce per far cadere i
campi ai giusti offset. In `GlobalUniformBufferObject` campi come `lightCount`,
`debugFlags`, `time` sfruttano il padding già presente, così `lights[]` resta
all'offset 64 e la struct non cresce.

---

## Pipeline e passi di rendering

**Pipeline** — la configurazione completa di un modo di disegnare: i due shader,
il formato dei vertici, blending, depth test, culling. Immutabile una volta
creata; ogni tecnica (`CookTorrance`, `Flame`, `Spectral`, `Shadow`...) ne ha
una.

**Render pass** — descrive un gruppo di attachment (i target su cui si disegna)
e cosa farne (pulire? conservare?). Il progetto ne ha diversi: scena principale
HDR, shadow map, i quattro pass di bloom.

**Subpass** — fase dentro un render pass. Le dipendenze tra subpass
(`ATDEP_SIMPLE`) sono le barriere che rendono visibili le scritture di un pass
alle letture del successivo.

**Attachment** — un'immagine su cui un render pass scrive: colore, profondità.
La scena disegna su un attachment `RGBA16F` (HDR), non direttamente sulla
swapchain.

**Framebuffer** — l'insieme concreto di immagini legate a un render pass per
un dato frame.

**Command buffer** — la lista registrata di comandi di disegno. Registrato
**una volta** per immagine della swapchain e rieseguito ogni frame: per questo
i dati che cambiano vanno in UBO (letti freschi a ogni draw), non in push
constant.

**Swapchain** — la coda di immagini che vengono mostrate a schermo a
rotazione. Di solito 2-3, da cui "2 frame in volo" e la necessità di un
command buffer / fence per immagine.

---

## Profondità e anti-aliasing

**Depth buffer (z-buffer)** — immagine che, per ogni pixel, conserva la
distanza dell'oggetto più vicino disegnato finora. Il depth test la confronta
prima di disegnare, così gli oggetti lontani non coprono i vicini.

**Depth test / depth write** — il confronto, e la scrittura del nuovo valore.
`Starter.hpp` forza `depthWriteEnable` su ogni pipeline: per questo ogni
fragment shader trasparente fa `discard` sotto una soglia di alpha, o un bordo
invisibile scriverebbe comunque profondità e bucherebbe ciò che sta dietro.

**Compare op (`LESS`, `LESS_OR_EQUAL`)** — la condizione del depth test.
`LESS_OR_EQUAL` serve dove due superfici stanno alla stessa profondità (le
quad sovrapposte della UI, i layer della fiamma) e con `LESS` la seconda
fallirebbe contro la profondità appena scritta dalla prima.

**Depth prepass** — un pass che scrive solo profondità, senza colore, per
fissare la superficie più vicina prima del pass di colore. `SpectralDepth.frag`
lo fa sui fantasmi così il lato lontano del fantasma non si fonde su quello
vicino.

**MSAA (Multi-Sample Anti-Aliasing)** — anti-aliasing che campiona la
copertura dei bordi più volte per pixel. Il progetto lo usa sul pass scena;
la cheat "shadow quality" lo attiva/disattiva.

**Resolve** — il passo che fonde i campioni MSAA in un'unica immagine normale.

---

## Colore, spazio lineare, HDR

**Spazio lineare vs sRGB** — le texture e lo schermo lavorano in sRGB (curva
percettiva), la matematica della luce va fatta in lineare. Un formato che
finisce in `_SRGB` fa la conversione in hardware: rifarla nello shader è un
bug (la "doppia gamma correction" di `notes.md`). **Regola: convertire una
volta sola.**

**Gamma** — l'esponente della curva sRGB (~2.2). "Gamma correction" =
applicare o togliere quella curva.

**HDR (High Dynamic Range)** — conservare valori di luce sopra 1.0 invece di
tagliarli a bianco. Il pass scena scrive su `RGBA16F` così il nucleo della
fiamma (~6x) e le scintille (~12x) sopravvivono fino al bloom.

**Tone mapping** — comprimere l'HDR nell'intervallo [0,1] mostrabile. Formula
del professore: `c / (Y(c) + 1)` con `Y` la luminanza. Spostato da
`CookTorrance.frag` a `Composite.frag` (ultimo pass) perché il bloom deve
prima trovare i pixel sopra 1.

**Luminanza (`Y`)** — la luminosità percepita di un colore, `dot(c, vec3(0.2126,
0.7152, 0.0722))` (Rec. 709). Dividere per la luminanza (non per canale) in tone
map mantiene la tinta invece di sbiadirla verso il bianco.

---

## Modello di illuminazione

**Equazione del rendering** — per un punto, la luce uscente è la somma su tutte
le sorgenti di (radianza in arrivo × BRDF). Il loop luci di `CookTorrance.frag`
è letteralmente questo.

**BRDF (Bidirectional Reflectance Distribution Function)** — la funzione che
dice quale frazione della luce che arriva da una direzione riparte verso
un'altra. Descrive il materiale.

**Radianza / irradianza** — radianza = luce lungo un raggio; irradianza = luce
che arriva su una superficie. Nel codice `lightRadiance()` restituisce la luce
in arrivo dopo decadimento e cono.

**Lambert (diffuso)** — modello del termine diffuso: la luce riflessa è
proporzionale a `cos θ = N·L`, uguale in tutte le direzioni. `clamp` a 0: una
faccia girata via non deve *sottrarre* luce.

**Blinn-Phong (superato nel progetto)** — specular basato sul *half vector*
`h = normalize(L + V)` e `pow(N·h, esponente)`. Sostituito da Cook-Torrance.

**Half vector (`h`)** — la bisettrice tra direzione della luce `L` e direzione
della vista `V`. I microfacet la cui normale è esattamente `h` sono gli unici
che portano luce da `L` a `V`.

**Cook-Torrance** — modello specular a microfacet: `mS · D·F·G / (4·(N·L)·(N·V))`.
Diffuso e specular sono **interpolati** da `k`, non sommati, o la superficie
restituirebbe più luce di quanta ne riceve.

**Microfacet** — l'idea che una superficie sia una folla di micro-specchi
perfetti orientati a caso; il modello non li disegna, ne descrive la statistica.

**D — distribution term (GGX)** — frazione di microfacet orientati lungo `h`.
È *il* riflesso: senza, nessun highlight. La `roughness` allarga il lobo. GGX
ha una "coda lunga" (nucleo stretto che sfuma in un alone ampio), realistica.

**G — geometric term** — i microfacet che si fanno ombra a vicenda. Senza,
le superfici ruvide esplodono di luce agli angoli radenti. Non ha parametri.

**F — Fresnel (Schlick)** — frazione riflessa invece che trasmessa, da `F0`
frontale fino a 1 all'orizzonte. È il motivo per cui ogni superficie opaca
diventa uno specchio se la guardi di taglio. `pow(1 - V·h, 5)` è il fit di
Schlick.

**F0** — riflettanza vista frontalmente. ~0.04 per *ogni* dielettrico (si
copia, non si sceglie); molto più alta e colorata per i metalli.

**roughness (ρ)** — quanto è ruvida la superficie, 0..1. 0 = specchio con
highlight minuscolo e nitido, 1 = gesso senza. Clampata a un minimo di 0.03
al caricamento (a 0 il denominatore GGX si annulla).

**k (quota diffusa)** — bilanciamento tra colore pieno e highlight. Il specular
prende `(1 - k)`. Per un metallo è forzata a 0.

**Albedo / `mD`** — il colore base della superficie, dalla texture, per pixel.
Non sta nel materiale.

**Specular color / `mS`** — colore dell'highlight. Bianco per i non-metalli
(l'highlight è il colore della *lampada*), tinto solo dai metalli. Per un
metallo diventa il colore di riflettanza del materiale.

**Dielettrico vs conduttore (metallico)** — un dielettrico lascia entrare la
luce che poi riesce diffusa; un metallo la assorbe (elettroni liberi), quindi
niente lobo diffuso e il termine indiretto è un *riflesso* della stanza
(`metalAmbient()`).

**Oren-Nayar** — modello diffuso alternativo per materiali retroriflettenti
(argilla, terra, stoffa). *Non* incluso: E06 prescrive Lambert per il diffuso
di Cook-Torrance.

---

## Normali

**Normale** — il vettore perpendicolare alla superficie in un punto. Da essa
dipende quasi tutta l'illuminazione.

**Normale al vertice vs normale di faccia** — la prima è memorizzata per
vertice e interpolata (superficie liscia); la seconda è costante su un
triangolo (superficie sfaccettata).

**Normal matrix (`nMat`)** — `inverse(transpose(mMat))`. Le normali non possono
cavalcare la matrice mondo come le posizioni: uno scale non uniforme (la
`road`, scalata `[1,4,1]`) le inclinerebbe fuori dalla superficie.

**Flat shading / `flatNormals`** — usare la normale di faccia. Flag per
materiale: i mesh MGCG hanno normali mediate sugli spigoli vivi, quindi una
scala di scalini viene sfumata invece che netta. Attivato solo dove serve
(non sulle torri, che sono cilindri sfaccettati e *vogliono* la media).

**Smooth shading** — usare le normali interpolate. Default del progetto.

---

## Luce indiretta e ambient

**Luce indiretta** — la luce che arriva dopo aver rimbalzato su altre
superfici. Le regole del progetto la rendono obbligatoria e considerano un
ambient costante appena sufficiente.

**Ambient hemisferico (E07)** — modello di luce indiretta che dipende dalla
normale: `mix(colore_terra, colore_cielo, (N·ambientDir + 1)/2)`. Una superficie
rivolta in su vede il cielo, una in giù vede il terreno, e sono colori diversi.

**Termine BRDF ambient** — la moltiplicazione per `mD`: la luce indiretta si
riflette col colore base come quella diretta.

**Ambient occlusion (AO)** — quanto è "chiusa"/incassata una superficie, che
riduce la luce indiretta che raccoglie. Il pack MGCG non ha mappe AO, quindi
il progetto usa un valore autorato per modello (`ambientWeight`,
`interiorAmbient`) come surrogato.

**`interiorAmbient`** — flag che blocca la miscela hemisferica a 0.5 (il peso
di una superficie verticale): dentro una stanza chiusa non c'è cielo sopra un
soffitto né cortile sotto.

**IBL (Image-Based Lighting) / environment map / split-sum** — usare una mappa
dell'ambiente prefiltrata come sorgente di luce indiretta. Non fatto (serve un
pass di cattura); `metalAmbient()` usa l'emisfero a due colori come surrogato
economico dello specular ambient split-sum.

**Blend E17** — la luce ambient come *quota* della luce del frame invece che
somma: `Lo·(1 - aw) + ambient·aw`. Sommandola, l'ambient sarebbe un pavimento
di luminosità sotto ogni pixel senza termine di visibilità.

---

## Tipi di luce

**Luce direzionale / diretta (il sole)** — nessuna posizione, nessun
decadimento: la sorgente è infinitamente lontana, ogni punto la vede dalla
stessa direzione.

**Luce puntiforme (point) — le torce** — ha una posizione, la direzione cambia
sulla superficie (per questo avvolge un barile), decade con la distanza.

**Luce spot** — una point moltiplicata per un termine a cono. Non è un modello
nuovo. Gli angoli del cono sono autorati come angoli *pieni* in gradi e
convertiti nel coseno del semiangolo al caricamento.

**Decadimento / attenuazione** — `(g / distanza)^β`. `g` = distanza a cui la
luce vale esattamente `color`. `β` non è il fisico 2: senza vera luce
indiretta l'inverso del quadrato risulta troppo scuro (le lanterne usano 1.4).

**Inverse-square (legge dell'inverso del quadrato)** — l'intensità fisicamente
corretta di una point light cala con `1/distanza²`.

**`NEAR_RADIUS`** — pavimento morbido sulla distanza efficace
(`sqrt(dist² + r²)`) così la radianza non diverge a ridosso della fiamma e la
curva si appiattisce vicino alla luce.

---

## Ombre (shadow mapping)

**Shadow mapping** — tecnica per le ombre: si renderizza la scena dal punto di
vista della luce salvando la profondità in una texture, poi in fase di
illuminazione si confronta ogni punto con quella texture per sapere se qualcosa
gli sta davanti rispetto alla luce.

**Shadow map 2D** — una singola texture di profondità prospettica/ortografica.
Usata per il sole (`Shadow.vert`/`.frag`).

**Cube shadow map** — sei facce (un cubo) attorno a una luce puntiforme,
campionate per *direzione* con un `samplerCube`. Ogni torcia ne ha una
(`ShadowCube.vert`/`.frag`).

**`samplerCube`** — sampler che si interroga con un vettore direzione invece
che con coordinate UV.

**Distanza lineare vs profondità proiettiva** — le cube map del progetto
salvano la distanza euclidea in world units, non la profondità 0..1: il lookup
per direzione non sa quale faccia ha risposto, quindi il valore deve significare
la stessa cosa su ogni faccia.

**Depth bias** — piccolo margine sottratto nel confronto per evitare che una
superficie si faccia ombra da sola. Nel progetto è ridotto al minimo (solo
rumore floating-point): l'errore vero è caricato altrove.

**Slope-scaled bias** — bias che cresce con l'inclinazione della superficie
vista dalla luce. Il progetto lo *evita* nella shadow map (renderebbe cieco
`LIGHT_DEBUG_SHADOW_GAP`) e cura l'acne a monte.

**Shadow acne** — auto-ombreggiatura a strisce dovuta alla quantizzazione della
shadow map. Risolta cullando le *front face* nel pass di cattura, così una
superficie illuminata non è nella sua stessa mappa.

**Peter-panning** — l'ombra che si stacca dalla base dell'oggetto, artefatto
di un bias o di un normal offset troppo grande.

**Normal offset** — spostare il punto di campionamento lungo la normale invece
di usare il bias. Provato e rimosso: costa uno spostamento laterale del bordo
dell'ombra.

**PCF (Percentage-Closer Filtering)** — ammorbidire il bordo dell'ombra
campionando la mappa in più punti attorno al lookup e mediando i verdetti.
`shadowFromCube()` fa 4 tap su una griglia ruotata.

**Front-face culling** — scartare le facce rivolte verso la camera. Nel pass
cube shadow scarta le *front face* (non le back), lasciando nella mappa solo il
lato lontano degli occlusori.

**Pool dinamico di ombre** — ci sono più luci "degne di ombra" che slot cube
(`NUM_SHADOW_CUBES`): `updateDynamicShadowSlots()` assegna gli slot alle luci
più vicine al giocatore, rivalutando nel tempo.

**`SHADOW_SWAP_MARGIN` / isteresi** — una luce in attesa deve battere
l'occupante di uno slot di un fattore, non di un pelo, per prenderne il posto:
altrimenti stando sul confine tra due candidate lo slot sfarfalla e ogni
scambio forza un re-render completo.

**Frustum** — il tronco di piramide di ciò che una camera (o una faccia di
shadow cube) vede. A 90° di FOV e aspetto 1:1 il confine delle sei facce è
letteralmente un cubo (da cui `DebugLines::PushBox`).

---

## Effetti procedurali (fuoco, fantasmi)

**Billboard** — una quad piatta tenuta sempre rivolta verso la camera. La
fiamma sono tre billboard: la silhouette è ritagliata per pixel nel fragment
shader, così può sfrangiarsi ai bordi come un numero fisso di vertici non
potrebbe.

**Procedurale** — generato da una formula invece che da una texture/mesh.

**Value noise** — rumore pseudo-casuale: si hasha ogni angolo di una cella
della griglia e si interpola con uno `smoothstep` (che nasconde la griglia).

**Hash** — funzione che da un input produce un numero pseudo-casuale
ripetibile. `hash21` (2D→1D), `hash11` (1D→1D, costruzione di Dave Hoskins).

**FBM (Fractional/Fractal Brownian Motion)** — somma di più *ottave* di value
noise, ognuna a frequenza doppia e ampiezza dimezzata: dettaglio a più scale
insieme (lingue larghe + bordi fini). Il `2.02` (non `2.0`) evita che le ottave
cadano sulla stessa griglia.

**Ottava** — un livello dell'FBM.

**Domain warp** — spostare il *punto di campionamento* di un rumore con un
secondo rumore a frequenza più bassa. È la differenza tra "lingue di fuoco che
si arricciano" e "un gradiente frizzante".

**Spina errante (wandering spine)** — la linea centrale della fiamma spostata
da rumore lento, nulla allo stoppino e massima in punta (`y·y`): fa arricciare
tutta la silhouette, non solo la luminosità.

**Inviluppo (envelope)** — il segnale lento (0.30..1.40) che governa la
luminosità della fiamma, simulato in CPU perché la luce puntiforme deve
pulsare sullo stesso segnale. Diviso in `intensity` (rapida, molla
criticamente smorzata) e `heightScale` (lenta, l'altezza segue in ritardo la
resa luminosa).

**Fresnel rim / termine di bordo** — luminosità/densità che sale dove la
superficie si gira via dalla vista (`pow(1 - N·V, potenza)`). Fa leggere un
guscio cavo come un volume; è tutto il contorno del fantasma (`Spectral.frag`).

---

## Post-processing

**Bloom** — l'alone luminoso attorno alle sorgenti molto brillanti. Il progetto
lo estrae dai pixel HDR sopra soglia e lo sfoca.

**Bright pass / soglia (`BLOOM_THRESHOLD = 1.55`)** — il pass che tiene solo i
pixel abbastanza luminosi da fioccare. Sopra 1.0, non a 1.0: una parete al sole
sfiora già 1.0 e a soglia 1.0 fioccherebbe tutto.

**Soft-knee** — rampa quadratica attorno alla soglia (Karis 2014) invece di un
taglio netto, così una scintilla che attraversa la soglia sfuma dentro invece
di sfarfallare.

**Karis average** — media pesata `1/(1+luma)` dei tap di downsample, così un
singolo pixel "firefly" (scintilla) non domina la media e non fa sfarfallare
la quad ridotta.

**Blur gaussiano separabile** — una sfocatura 2D fatta come due passate 1D
(orizzontale poi verticale): `2N` tap invece di `N²`. Kernel a 9 tap.

**Downsample** — ridurre la risoluzione. Il bloom gira a un quarto di
risoluzione: un kernel di larghezza fissa copre quattro volte più immagine, da
cui un alone ampio a basso costo.

**Composite** — l'ultimo pass: somma scena + bloom, applica esposizione, tone
map, grading, vignette, dither, scrive sulla swapchain.

**Vignette** — angoli dello schermo scuriti. Coppia con la fog di distanza:
la fog nasconde la distanza *davanti*, la vignette i *bordi*.

**Chromatic aberration** — canali R e B tirati lungo la direzione radiale ai
bordi, come l'aberrazione di una lente reale. Il canale G resta al suo posto
così il mondo resta allineato al mirino.

**Split-toning / grading** — ombre leggermente fredde, alte luci leggermente
calde, pesato per luminanza. Fa leggere la luce di torcia più calda sulla
pietra senza toccare l'illuminazione.

**Dithering** — un LSB a 8 bit di rumore triangolare aggiunto per ultimo, che
rompe le bande nei gradienti scuri prima che la swapchain li quantizzi.

**Fog esponenziale al quadrato** — `exp(-(densità·dist)²)`. Resta vicino a 1
presso la camera, dove la chiarezza serve al gameplay, e morde nella metà
lontana, dove serve al culling geometrico.

---

## Trasparenza

**Alpha blending** — comporre un pixel semitrasparente su ciò che sta sotto:
`srcAlpha·src + (1 - srcAlpha)·dst`.

**Premultiplied alpha** — quando il colore è già moltiplicato per l'alpha. Gli
shader del progetto **non** lo sono: moltiplicare il colore per l'alpha nello
shader lo eleverebbe al quadrato.

**`discard`** — scartare un fragment prima che scriva colore *o* profondità.
Necessario (non ottimizzazione) con `depthWriteEnable` forzato: un bordo
alpha≈0 non scartato bucherebbe ciò che sta dietro.

---

## Spazi e matrici

**Object / local space** — le coordinate come stanno nel file del modello. I
mesh MGCG sono Z-up con origine alla base, quindi il "su" locale è `-Z`.

**World space** — coordinate della scena, dopo la matrice mondo (`mMat` / `Wm`)
dell'istanza.

**View space** — coordinate relative alla camera.

**Clip space / NDC (Normalized Device Coordinates)** — dopo la proiezione;
NDC va da -1..1 (in Vulkan la profondità 0..1, con
`GLM_FORCE_DEPTH_ZERO_TO_ONE`). Le quad full-screen (`Post.vert`) hanno i
vertici già in NDC.

**Matrice MVP** — `Projection · View · Model`. Porta un vertice da object space
a clip space in un colpo. `mvpMat` nell'UBO.

**View-projection (`vpMat` / ViewPrj)** — `Projection · View`, senza il modello.
Usata da `DebugLines` e per costruire le matrici di shadow.

**glTF** — il formato dei modelli 3D. `Starter.hpp` accetta solo glTF ASCII;
per questo `tools/convert_assets.py` converte il pack Dracula.

**UV / texture atlas** — le coordinate 2D che mappano un pixel del modello a un
punto della texture. Le texture Dracula sono atlas UV (non piastrellabili).

---

## Collisioni e camera

**AABB (Axis-Aligned Bounding Box)** — scatola allineata agli assi mondo.
`"collider": "AABB"` in `scene.json` ne adatta *una* a tutto il modello:
giusto per mesh a scatola, sbagliato per cose cave o profilate (il cancello,
la scala), che ottengono scatole autorate in `colliders.json`.

**OOBB / OBB (Oriented Bounding Box)** — scatola ruotata. `getExtents()`
restituisce comunque il suo inviluppo AABB, quindi una superficie inclinata
non sopravvive: le rampe hanno una lista propria, valutate in spazio
model-local dove restano esatte a qualsiasi angolo.

**BVH (Bounding Volume Hierarchy)** — albero di volumi contenitori.
Deliberatamente non usato: il nodo padre finisce nella lista di gameplay
accanto ai figli e il suo `getExtents()` è l'unione di tutti (la scatola che
riempie l'arco, di nuovo).

**Ground clamp** — il passo verticale che aggancia i piedi alla superficie
calpestabile più alta.

**Step height (`MAX_STEP_HEIGHT`)** — l'altezza massima di un gradino su cui si
sale automaticamente. Il passo muri e il passo terra devono concordare su cosa
sia "un gradino" o si annullano a vicenda.

**Vertical view smoothing (`eyeStepOffset`)** — assorbe lo scatto della camera
quando si sale un gradino e decade a zero con costante di tempo 0.06s. Solo
gli scatti *verso l'alto* (attenuare le cadute farebbe sembrare la gravità
gommosa).

**Culling** — non disegnare/processare ciò che non conta. *Frustum culling*:
fuori dalla vista. *Distance culling* (`GEOM_CULL_*`, `TORCH_LIGHT_CULL_DIST`):
oltre una distanza dalla camera.

**Cheat / debug flag** — i toggle in `CheatFlags`: nessuna persistenza, si
resettano ai default "onesti" a ogni avvio. Le nove cheat luce servono a
distinguere i casi in cui una parete nera è una luce morta, una normale
girata, un materiale che mangia tutto o una texture scura.

---

## Segnali e filtri

**Molla criticamente smorzata (critically-damped spring)** — insegue un
bersaglio senza mai superarlo (nessun overshoot) e senza oscillare. Smussa
`intensity` della fiamma.

**Filtro passa-basso / costante di tempo (τ, tau)** — lascia passare i
cambiamenti lenti e taglia quelli rapidi. `FLAME_HEIGHT_TAU = 0.35s` fa seguire
l'altezza della fiamma in ritardo alla sua resa luminosa.

**Edge-triggered (rilevamento di fronte)** — reagire *una volta* alla pressione
di un tasto, non a ogni frame in cui è tenuto premuto. Idioma
`xKeyWasPressed` in tutti i menu.
