# Come funziona il progetto

Guida completa a tutti gli elementi del progetto, scritta partendo da zero.

Come è organizzata: la **Parte 0** spiega i concetti di base (cos'è uno shader,
cos'è un uniform buffer, cos'è un render pass). Se sai già queste cose, saltala.
Dalla Parte 1 in poi c'è il progetto vero, e ogni sezione ha in fondo un blocco
**Se il prof chiede** con le domande probabili e la risposta.

Complementare a `notes.md`, che è il diario delle decisioni (*perché* abbiamo
fatto una scelta e cosa abbiamo provato prima). Questo file spiega invece *come
funziona adesso*. `GLOSSARY.md` raccoglie le definizioni brevi dei termini di
Computer Graphics usati qui.

Dove sta il codice, tutto sotto `skeleton/source/`:

- `src/main.cpp` — la classe `Castlescape : BaseProject`, ~6500 righe: le struct degli uniform buffer, lo stato di gioco, il grafo di rendering, la logica.
- `src/Libs.cpp` — definisce le macro `*_IMPLEMENTATION` una volta sola (i moduli in `custom/` sono header-only: dichiarazione e implementazione nello stesso file, e serve un solo punto in cui l'implementazione viene davvero compilata).
- `include/modules/` — il framework del professore (`Starter.hpp`, `Scene.hpp`, `TextMaker.hpp`, `Animations.hpp`, `Colliders.hpp`). Non lo tocchiamo: il prof valuta usando la sua copia, quindi ogni nostra modifica lì sparirebbe.
- `include/custom/` — i moduli scritti da noi.
- `shaders/` — codice GLSL, compilato in SPIR-V da CMake al momento della build. Una cartella per lavoro: `scene/` (la coppia che disegna la scena), `spectral/` (i fantasmi), `shadow/`, `post/` (bloom e composite), `fire/` (fiamme e scintille), `exit/` (il bagliore dell'uscita), `ui/`, `debug/`, `framework/` (gli shader del framework del prof).
- `assets/scenes/*.json` — i dati: scena, materiali, luci, fiamme, collider, gameplay.

---

# PARTE 0 — Le basi

## 0.1 Cosa fa davvero una GPU

Il modello mentale giusto: la GPU è una macchina che sa fare **una cosa sola**,
molto bene e milioni di volte in parallelo — trasformare triangoli in pixel
colorati. Il processo si chiama **pipeline grafica** ed è a stadi fissi, tranne
due punti in cui il programmatore inserisce codice proprio.

Il flusso, semplificato:

1. **Input assembly** — la GPU legge un buffer di vertici e un buffer di indici, e li raggruppa in triangoli.
2. **Vertex shader** — codice nostro. Gira **una volta per vertice**. Il suo compito obbligatorio è scrivere `gl_Position`, cioè dove finisce quel vertice sullo schermo.
3. **Rasterizzazione** — stadio fisso: la GPU guarda i tre vertici del triangolo, capisce quali pixel dello schermo copre, e per ciascuno **interpola** i valori che il vertex shader ha prodotto. Se i tre vertici hanno normali diverse, il pixel al centro riceve la media pesata delle tre.
4. **Fragment shader** — codice nostro. Gira **una volta per pixel coperto** (un "fragment" è un candidato pixel). Deve scrivere un colore.
5. **Depth test e blending** — stadio fisso: il fragment viene scartato se qualcosa di più vicino l'ha già scritto, altrimenti va nel framebuffer.

Numeri che rendono l'idea: in questo progetto il vertex shader gira qualche
decina di migliaia di volte per frame, il fragment shader qualche milione. È
il motivo per cui tutto ciò che si può calcolare per vertice si calcola per
vertice, e tutto ciò che si può calcolare una volta per frame sulla CPU si
calcola sulla CPU.

## 0.2 Come arrivano i dati dentro uno shader

Uno shader è una funzione pura: non può aprire file, non può leggere la RAM di
sistema, non ha variabili globali persistenti. Tutto ciò che sa gli deve essere
passato. Ci sono esattamente quattro canali.

**1. Attributi di vertice** (`layout(location = N) in`). Dati che cambiano *da
vertice a vertice*: posizione, normale, coordinate texture. Vivono in un vertex
buffer. Nel nostro progetto il formato si chiama `VDposNormUV` e ogni vertice è
`vec3 pos, vec3 norm, vec2 UV` = 32 byte.

**2. Uniform buffer** (`layout(binding=..., set=...) uniform Blocco {...}`). Un
blocco di memoria **costante per tutta la durata di una draw call**, ma
riscrivibile dalla CPU fra una draw e l'altra. "Uniform" significa esattamente
questo: uguale per ogni vertice e ogni pixel di quel draw. È il canale
principale, e la Parte 2 di questo documento parla solo di questo.

**3. Texture** (`uniform sampler2D`, `samplerCube`). Immagini campionabili. Il
`sampler` è l'oggetto che decide *come* si legge l'immagine: filtraggio (nearest
o bilineare), e cosa succede se chiedi una coordinata fuori dai bordi (repeat,
clamp, ...). Questo dettaglio ci morde in almeno tre punti del progetto, vedi
§5.9.

**4. Push constant** (`layout(push_constant) uniform ...`). Pochi byte (il minimo
garantito da Vulkan è 128) che vengono scritti **direttamente dentro il command
buffer** insieme al comando di draw. Sono velocissimi perché non passano da un
buffer in memoria. Hanno però un difetto enorme in questo progetto, vedi §2.1.

Il flusso è sempre lo stesso: CPU scrive → GPU legge. Non c'è mai ritorno, gli
shader non possono rispondere.

## 0.3 Descriptor set: il concetto che confonde tutti

Un uniform buffer o una texture non si "passano" a uno shader come argomenti di
funzione. Vulkan usa un'indirezione in due passi.

Un **descrittore** (descriptor) è un puntatore tipizzato a una risorsa: "il
buffer che sta a questo indirizzo, lungo 240 byte" oppure "questa immagine con
questo sampler".

Un **descriptor set** è un gruppetto di descrittori bindati insieme. Uno shader
li vede come `set = 0`, `set = 1`, `set = 2`, e dentro ciascuno i singoli
descrittori come `binding = 0`, `binding = 1`, ...

Il **descriptor set layout** è lo schema: dice "il set 1 ha un uniform buffer al
binding 0 e una texture al binding 1". Serve alla pipeline per sapere che forma
avranno i set che le verranno bindati, e va dichiarato **prima**, alla creazione
della pipeline.

Perché più set invece di uno solo? Per **frequenza di aggiornamento**. Bindare
un set costa; se metti tutto insieme, per cambiare una cosa devi ri-bindare
tutto. La convenzione (che questo progetto segue alla lettera) è:

- **set 0** — cose che cambiano una volta per frame (camera, luci).
- **set 1** — cose che cambiano una volta per oggetto (matrici, materiale, texture).
- **set 2** — qui: le shadow map, che non stanno né in "per frame" né in "per oggetto" ma in "per luce che proietta ombre".

I descriptor set vengono allocati da un **descriptor pool**, la cui dimensione
va dichiarata *prima* di allocarne anche uno. Da qui i conteggi in
`DPSZs.uniformBlocksInPool` / `texturesInPool` / `setsInPool` in `localInit()`:
se sbagli il conto, l'allocazione fallisce a runtime.

## 0.4 Pipeline

Una **pipeline** in Vulkan è un oggetto immutabile che congela *tutto* lo stato
del disegno: quale vertex shader, quale fragment shader, il formato dei vertici,
i descriptor set layout, il culling, il depth test, il blending, il numero di
sample MSAA. Cambiare anche una sola di queste cose vuol dire creare una
pipeline nuova.

Per questo il progetto ha otto pipeline: `P` (la scena), <mark>`PShadow`,</mark>
`PShadowCube`, `Pbright`, `PblurH`, `PblurV`, `Pcomposite`, più quelle di
fiamma, ExitGlow, UiQuad, testo. Non è ridondanza: sono davvero configurazioni
diverse.

Una pipeline si crea contro un **render pass**, e vale la regola di
**compatibilità**: una pipeline creata contro un render pass funziona con
qualunque altro render pass che abbia la stessa configurazione di attachment. È
il motivo per cui abbiamo *una* `PShadow` condivisa da tutti gli shadow pass 2D
invece di una per pass.

## 0.5 Render pass, attachment, framebuffer

Un **attachment** è un'immagine su cui si disegna: il colore, la profondità,
ecc. Un **framebuffer** è un insieme di attachment. Un **render pass** descrive
cosa succede a quegli attachment durante il disegno: si puliscono all'inizio
(`LOAD_OP_CLEAR`)? Il contenuto va conservato alla fine (`STORE_OP_STORE`)? In
quale layout di memoria si trova l'immagine prima e dopo?

Il concetto di **image layout** è specifico di Vulkan: la stessa immagine, in
memoria, è organizzata diversamente a seconda di cosa ci devi fare. Un layout
ottimale per scriverci come color attachment non è quello ottimale per leggerla
come texture. Il passaggio da un layout all'altro è una **transizione**, e
costa; il render pass la fa per te se gliela dichiari.

Le **subpass dependency** sono la parte più ostica. La GPU esegue i comandi in
modo massicciamente parallelo e fuori ordine: se il pass A scrive un'immagine e
il pass B la legge, **nulla garantisce** che A abbia finito quando B comincia, a
meno che tu non lo dichiari. Una dependency dice: "prima che lo stadio X del
pass B possa fare l'accesso Y, lo stadio Z del pass A deve aver finito
l'accesso W".

Nel progetto questo compare in §1 (l'ordine della catena di post-processing) e
in §5 (le ombre devono essere finite prima che la scena le campioni).

## 0.6 Command buffer e swapchain

Non si comanda la GPU una chiamata alla volta: si **registra** una lista di
comandi in un command buffer e poi la si **submitta** tutta insieme.

La **swapchain** è la coda di immagini che finiscono a schermo — tipicamente 2 o
3, così mentre una viene mostrata la GPU disegna già sulla successiva. Da qui
l'espressione "frame in volo": ci sono più frame contemporaneamente in
lavorazione.

Questo ha una conseguenza pesantissima su tutto il progetto: `Starter.hpp`
registra il command buffer "main" **una volta sola per ogni immagine della
swapchain**, e poi lo riesegue identico per sempre. Non lo ri-registra a ogni
frame. È efficiente, ma significa che tutto ciò che è *scritto dentro* il
command buffer è congelato al momento della registrazione. Vedi §2.1: è la
ragione per cui quasi niente in questo progetto è una push constant.

Significa anche l'inverso: qualunque cosa registri nel main buffer si
ridisegnerà a ogni frame per sempre, anche se il suo contenuto non cambia mai.
Vedi §5.5, il caching delle shadow map.

## 0.7 Depth buffer

Il **depth buffer** (o z-buffer) è un'immagine parallela a quella di colore che
memorizza, per ogni pixel, la profondità di ciò che è stato scritto lì. Prima di
scrivere un fragment, la GPU confronta la sua profondità con quella già
memorizzata: se è più lontano, lo butta. È così che gli oggetti si occludono a
vicenda senza doverli ordinare.

Due parole che tornano spesso:

- **depth test** — il confronto. Si può disattivare.
- **depth write** — la scrittura del nuovo valore. Anche questa si può disattivare, e per gli oggetti trasparenti **si dovrebbe**: un vetro non deve impedire di vedere quello che c'è dietro.

`Starter.hpp` hardcoda `depthWriteEnable = VK_TRUE` su **ogni** pipeline,
trasparenti incluse, e noi non possiamo toccarlo. È il motivo per cui
`Flame.frag`, `Spark.frag` ed `ExitGlow.frag` usano `discard` invece di scrivere
alpha basso: se non scartassero il frammento, questo scriverebbe profondità e
bucherebbe tutto quello che sta dietro con la propria sagoma. Il `discard` lì non
è un'ottimizzazione, è **necessario per la correttezza**.

## 0.8 Lineare, sRGB, HDR e tone mapping

Questo è il blocco concettuale che spiega metà delle scelte del progetto.

**La luce si somma in modo lineare.** Se due lampade illuminano un muro, la
luminosità è la somma delle due. Tutti i calcoli di illuminazione devono quindi
avvenire in **spazio lineare**.

**Il monitor non è lineare.** Un valore 0.5 in un file immagine non viene
mostrato a metà luminosità, ma a circa il 21%: c'è di mezzo una curva, lo
standard **sRGB**. Storicamente è così perché l'occhio umano è più sensibile alle
differenze nelle zone scure, quindi conviene dedicare più bit lì.

Conseguenza: bisogna **decodificare** le texture da sRGB a lineare quando le si
legge, e **ri-codificare** il risultato prima di mostrarlo. Se lo si fa due
volte (o zero volte) l'immagine viene sbagliata. Nel progetto:

- in lettura, la image view della texture è `VK_FORMAT_R8G8B8A8_SRGB`, quindi la decodifica la fa **l'hardware del sampler**. Per questo `CookTorrance.frag` legge l'albedo e la usa direttamente, senza `pow(c, 2.2)`.
- in scrittura, la swapchain è `B8G8R8A8_SRGB`, quindi la codifica la fa **l'hardware**. Per questo `Composite.frag` scrive lineare. Una curva a mano lì raddoppierebbe la correzione — ed è esattamente il bug ereditato dallo skeleton, documentato in `notes.md`.

**HDR** (High Dynamic Range) significa semplicemente che i valori di colore non
sono limitati a [0,1]. Una fiamma può valere 6, una scintilla 12. Un normale
attachment a 8 bit per canale non può rappresentarlo: 1.0 è il massimo, tutto
sopra viene tagliato. Per questo la scena viene disegnata in un attachment
**RGBA16F** (16 bit float per canale), che regge valori arbitrari.

**Tone mapping** è la funzione che alla fine schiaccia quell'intervallo
illimitato dentro [0,1] per il monitor. Qui usiamo `c / (Y + 1)`, dove `Y` è la
luminanza: si divide per la luminanza e non canale per canale, perché dividendo
canale per canale le alte luci si desaturano verso il bianco.

**E il bloom?** Il bloom è l'alone attorno alle sorgenti molto luminose (nella
realtà è diffusione dentro l'occhio e dentro l'obiettivo). Si ottiene cercando i
pixel *sopra 1*, sfocandoli e ri-aggiungendoli. Da cui il punto cruciale: **il
tone map deve venire dopo il bloom**, altrimenti quando il bloom va a cercare i
pixel sopra 1 non ne trova più nessuno. È esattamente per questo che il tone map
è stato spostato da `CookTorrance.frag` a `Composite.frag`.

## 0.9 Il framework del professore

`Starter.hpp` fornisce `BaseProject`, che gestisce inizializzazione di Vulkan,
swapchain, loop principale, e chiama i nostri hook: `setWindowParameters()`,
`localInit()`, `pipelinesAndDescriptorSetsInit()`, `populateCommandBuffer()`,
`updateUniformBuffer()`, `localCleanup()`. Più le classi `Model`, `Texture`,
`Pipeline`, `RenderPass`, `DescriptorSet`, `DescriptorSetLayout`.

`Scene.hpp` legge `scene.json` e costruisce modelli, texture e istanze,
raggruppate per **tecnica** (una tecnica = una pipeline più i suoi descriptor
set). Parsa però solo `id`, `model`, `texture`, `translate`, `eulerAngles`,
`scale`: **è la ragione per cui esistono tutti gli altri nostri file JSON**.
Materiali, luci, collider e gameplay non hanno posto in `scene.json` e
`Scene.hpp` non si può estendere.

`Colliders.hpp` costruisce AABB automatiche, `TextMaker.hpp` disegna testo da un
atlante di font, `Animations.hpp` gestisce animazioni.

---

# PARTE 1 — Il grafo di rendering

Un frame intero, dall'inizio alla fine. Tutto ciò che segue è registrato
nell'unico command buffer "main" (`populateCommandBuffer()`, main.cpp:4130),
in quest'ordine:

1. **Shadow pass 2D** — un render pass depth-only per ogni luce con shadow map 2D (`NUM_SHADOW_MAPS_2D = 2`, oggi la usa solo il sole). Pipeline `PShadow` (`Shadow.vert` + `Shadow.frag`).
2. **Shadow pass cubemap della torcia in mano** — 6 facce, pipeline `PShadowCube` (`ShadowCube.vert` + `ShadowCube.frag`). Solo questa torcia: tutte le altre cubemap sono renderizzate fuori dal command buffer, vedi §5.5.
3. **Scene pass** (`RP`) — verso un attachment **HDR offscreen RGBA16F**, non verso lo schermo. Dentro, nell'ordine: geometria opaca e depth prepass dei fantasmi (`Scene::populateCommandBuffer`), colore dei fantasmi (emesso a mano, §3.14), fiamme + scintille, ExitGlow, LightDebug.
4. **Bright pass** (`RPbright`, a un quarto di risoluzione) — soglia + downsample, `BloomBright.frag`.
5. **Blur H** (`RPblurH`) — gaussiana 1D orizzontale, `BloomBlur.frag`.
6. **Blur V** (`RPblurV`) — la stessa shader, con `blurDir` = (0,1).
7. **Composite** (`RPcomposite`) — verso la swapchain: scena + bloom, esposizione, tone map, whiteout finale. `Composite.frag`.

Il testo (`TextMaker`) e la HUD (`UiQuad`) sono command buffer separati,
submittati dopo (submit order 10000 e 9000 contro lo 0 del main), quindi
finiscono sopra al frame già composto.

**Perché l'ordine è rispettato.** Non ci sono barriere esplicite: l'ordine è
imposto dalle **subpass dependency**. Ogni pass offscreen usa `ATDEP_SIMPLE`, la
cui seconda dependency (subpass 0 → EXTERNAL, `COLOR_ATTACHMENT_OUTPUT` →
`FRAGMENT_SHADER`) è esattamente la barriera che rende le scritture di un pass
visibili alla lettura come texture del pass successivo.

La *prima* dependency conta altrettanto ed è più facile da dimenticare: è una
write-after-read contro il frame **precedente**. Ci sono due frame in volo e una
sola immagine per ogni attachment offscreen, quindi senza quella il frame N+1
sovrascriverebbe un target mentre il frame N lo sta ancora campionando.

**MSAA.** `msaaSamples = VK_SAMPLE_COUNT_4_BIT`, forzato in `localInit()` prima
che `RP.init()` lo legga. MSAA normalmente valuta il fragment shader una volta
per pixel e fa il test di copertura per sample; `Starter.hpp` però attiva
`sampleShadingEnable = VK_TRUE` con `minSampleShading = 1.0f`, che lo trasforma
in **supersampling**: il fragment shader gira una volta per *sample*. Il default
del framework è `getMaxUsableSampleCount()`, cioè 16 su questa macchina:
`CookTorrance.frag` girava 16 volte per pixel, ogni volta con il loop completo
sulle luci e la valutazione GGX. Misurato sulla Iris Xe: 25 FPS a 16 sample, 94 a
4, 175+ a 2. Quattro è il compromesso.

Assegnarlo è lecito senza toccare `Starter.hpp`: `msaaSamples` è un membro
protected di `BaseProject` e `pickPhysicalDevice()` (che imposta il default)
gira prima di `localInit()`.

**Risoluzione interna (`renderScale`, oggi 0.8).** La scena 3D — `RP` e la
catena HDR in cui disegna `CookTorrance.frag` — è renderizzata a questa frazione
della risoluzione della finestra **su ogni asse**, cioè a `renderScale²` dei
pixel: 0.8 è circa il 64%. `Composite.frag` la riporta poi alla dimensione reale
della finestra tramite lo stesso sampler bilineare con cui già legge `srcTex`,
quindi l'output riempie comunque la finestra a piena risoluzione: solo il
dettaglio della scena è calcolato su meno pixel.

Il guadagno è proporzionale al taglio di pixel, e si compone con il punto sopra:
ogni invocazione di fragment risparmiata è una su cui il per-sample shading
avrebbe fatto girare il loop completo sulle luci. Si paga in una scena
leggermente più morbida, che nebbia, vignette, bloom e il buio generale di un
dungeon nascondono bene.

**La UI non è inclusa**: testo, prompt, crosshair, HUD e menu girano in pass e
command buffer propri, sottomessi dopo `RPcomposite`, e restano
incondizionatamente alla risoluzione vera della finestra. È il motivo per cui il
testo resta nitido qualunque valore abbia lo slider.

Sia `renderScale` che il livello MSAA sono **membri runtime e non costanti di
compilazione**, perché entrambi hanno uno slider nel menu. Cambiarli rigioca lo
stesso percorso di ricostruzione di un resize vero (`onWindowResize()` alla
dimensione corrente, poi `RebuildPipeline()`). Il livello MSAA è tenuto come
log2 (0 → 1×, 1 → 2×, 2 → 4×) perché lo slider ha passo additivo fisso di ±1 e
i sample count Vulkan devono essere potenze di due; il massimo viene da
`getMaxUsableSampleCount()`, così una GPU che non regge 16× non se lo vede
offrire.

Una nota di sincronizzazione che vale la pena saper spiegare: l'applicazione è
**saltata se una ricostruzione è già pendente**. `recreateSwapChain()` gira una
volta sola, in fondo al `drawFrame()` del frame corrente; impilarci sopra una
seconda dimensione bersaglio prima che sia avvenuta è ciò che una volta ha fatto
iniziare un render pass contro una dimensione più nuova del framebuffer a cui
era legato — gli errori di validazione "renderArea greater than framebuffer", e
il crash subito dopo.

> **Se il prof chiede**
>
> *"Perché la scena non disegna direttamente sullo schermo?"* — Perché serve un
> target floating point in cui i valori possano superare 1.0, altrimenti il
> bright pass del bloom non avrebbe niente da trovare (§0.8).
>
> *"Come garantisci che il blur legga il bright pass già finito?"* — Con le
> subpass dependency del render pass, non con barriere manuali. `ATDEP_SIMPLE`
> ordina `COLOR_ATTACHMENT_OUTPUT` del pass precedente contro `FRAGMENT_SHADER`
> del successivo.
>
> *"Perché 4 sample e non il massimo?"* — Perché il framework forza il per-sample
> shading, quindi MSAA qui è supersampling e il costo è lineare nel numero di
> sample. Numeri misurati sopra.

---

# PARTE 2 — Gli Uniform Buffer Object

La parte che il prof quasi certamente approfondirà, perché è dove si vede se hai
capito Vulkan o hai solo copiato lo skeleton.

La regola da tenere sempre in testa: **un UBO è un blocco di memoria mappato
dalla CPU e letto dalla GPU al momento del draw, non al momento della
registrazione del command buffer.** Da qui discende quasi tutto.

## 2.1 Perché quasi tutto è UBO e non push constant

Come detto in §0.6, `BaseProject::submitCommandBuffer()` registra il command
buffer "main" **una volta per immagine della swapchain** e poi lo riesegue
identico ogni frame.

Una push constant viene scritta *dentro* il command buffer con
`vkCmdPushConstants`. Se il buffer viene registrato una volta e riprodotto per
sempre, quel valore è **congelato al momento della registrazione**.

Quindi:

- **Va bene come push constant** tutto ciò che è determinato da *dove* sta il draw dentro il buffer registrato, non da quando viene eseguito. Due esempi nel progetto: la faccia della cubemap (`ShadowCubeFacePushConstant` — quella iterazione del loop è sempre la faccia 0, poi sempre la 1, ecc.) e la view-projection del sole in `Shadow.vert` (il sole non si muove mai).
- **Non va bene** niente che cambi frame per frame: la posizione della torcia in mano, le matrici delle cubemap dinamiche. Quelle stanno in un uniform buffer ri-mappato ogni frame in `updateUniformBuffer()`.

Il punto sottile: il *contenuto* di un uniform buffer viene letto fresco al
momento del draw, indipendentemente da quando è stato registrato il comando che
lo binda. È la stessa ragione per cui la matrice di mondo di ogni istanza
sopravvive all'essere ri-mappata ogni frame senza ri-registrare niente.

**Corollario meno ovvio:** anche gli UBO che non cambiano mai vengono ri-mappati
ogni frame. `map()` scrive nello slot del buffer corrispondente all'**immagine
corrente della swapchain**; mappare solo lo slot 0 lascerebbe gli altri con
quello che c'era in memoria all'allocazione. Il sole non si muove, ma
`ShadowUniformBufferObject` viene comunque riscritto ogni frame per questo.

## 2.2 std140: le regole di allineamento

Un uniform block usa il layout **std140**, che è uno standard di impaginazione
della memoria. Le regole che servono qui:

- uno scalare (`float`, `int`) si allinea a 4 byte e ne occupa 4;
- un `vec2` si allinea a 8;
- un `vec3` **si allinea a 16 e riserva uno slot da 16**, pur usandone solo 12;
- un `vec4` e una `mat4` si allineano a 16;
- una struct dentro un array viene arrotondata a un multiplo di 16.

La conseguenza pratica che il progetto sfrutta ovunque: **uno scalare messo
subito dopo un `vec3` è gratis**, perché occupa i 4 byte di padding che std140
metterebbe comunque. Da qui il pattern ricorrente `vec3 eyePos; int lightCount;`,
`vec3 pos; float g;`, `vec3 color; float cosIn;`.

Il problema vero è che **C++ e GLSL devono concordare byte per byte**. Il
compilatore C++ ha le sue regole di packing, che non sono std140. Se non
coincidono, lo shader legge i campi agli offset sbagliati e il risultato è
silenzioso e incomprensibile (materiali sbagliati, matrici a caso). Si forza la
coincidenza con `alignas(16)` sui `vec3` e sulle `mat4` nella struct C++.

Dove non serve niente: `PostUniformBufferObject` è due `vec2` seguiti da sei
scalari, e la disposizione naturale del C++ coincide già con std140.

Una nota tecnica che vale la pena saper spiegare: `nMat` è dichiarata `mat4`
anche se matematicamente è una `mat3`. Motivo: std140 tratta una `mat3` come un
array di 3 `vec3`, quindi ne padda ogni colonna a 16 byte — 48 byte invece dei 36
"logici", con regole facili da sbagliare. Una `mat4` è meno ambigua e costa 16
byte in più.

## 2.3 `UniformBufferObject` — set 1, per-istanza (240 byte)

Definito a main.cpp:50. È il blocco più importante, ed è letto da **quattro
shader diversi**: `PosNormUV.vert` e `CookTorrance.frag` (a set 1), `Shadow.vert`
e `ShadowCube.vert` (a set 0, perché quelle pipeline non hanno il `DSLglobal`
davanti e quindi la numerazione dei set scala di uno).

Che i quattro condividano lo stesso buffer non è un dettaglio estetico: è quello
che fa sì che un occlusore in movimento (una porta che si apre, un fantasma)
proietti un'ombra che lo segue. Lo shadow pass legge la **stessa** `mMat` che il
main pass ha appena aggiornato, senza che nessuno debba tenere in sync una
seconda copia.

Campi:

- `mat4 mvpMat` — model-view-projection: la matrice che porta un vertice da spazio modello a clip space.
- `mat4 mMat` — la sola matrice di mondo. È l'unico campo letto dai due shader di shadow.
- `mat4 nMat` — inversa-trasposta di `mMat`, per trasformare le normali (vedi §3.1 per il perché non basta `mMat`).
- `vec3 mS` + `float roughness` — colore speculare e larghezza del lobo GGX.
- `float F0`, `float k`, `int flatNormals`, `int interiorAmbient`.
- `float time`, `float ambientWeight`, `float glow`, `int metallic`.

I dodici byte-slot finali cadono negli slot std140 `[mS.xyz | roughness]`,
`[F0 | k | flatNormals | interiorAmbient]`, `[time | ambientWeight | glow |
metallic]`: 192 (tre mat4) + 48 = **240 byte**, identici in C++ e in GLSL.
`glow` e `metallic` sono stati aggiunti nello spazio libero del terzo slot, ed è
per questo che la dimensione non si è mossa quando sono comparsi.

Il colore base (`mD`, l'albedo) **non è qui**: viene dalla texture, per pixel.

Note sui singoli campi:

- `flatNormals` — le mesh MGCG mediano le normali dei vertici anche sugli spigoli vivi. Una faccia piatta esce quindi con un gradiente di illuminazione invece che con un valore costante (E06 s.3-16: un solido a spigoli vivi vorrebbe i vertici duplicati per faccia, e queste mesh non li hanno). Con questo flag il fragment shader si ricava la normale della faccia da solo, con `cross(dFdx(fragPos), dFdy(fragPos))`: le derivate della posizione di mondo lungo il triangolo sono due vettori nel suo piano, quindi il loro prodotto vettoriale è perpendicolare al piano. Il segno dipende dall'ordine di avvolgimento, quindi viene orientato contro la normale interpolata.
- `interiorAmbient` — blocca il blend dell'ambient emisferico a 0.5, cioè al valore che avrebbe una parete verticale, invece di ricavarlo dalla normale. <mark>In interni "cielo" e "terra" non esistono.</mark> Prendere alla lettera l'estremo "terra" faceva raccogliere a un soffitto di dungeon solo `ambientLower`: 1.80 volte più scuro delle pareti che tocca, e marrone dove quelle sono fredde.
- `time` — secondi dall'avvio. Viaggia qui invece che nel global UBO per non spostare l'offset di `lights[]`. Lo leggono solo le shader delle fiamme e il blocco del focus glow; gli altri lo dichiarano e lo ignorano, perché le due pipeline condividono `DSLlocal` e quindi questa unica struct.
- `glow` — <mark>0..1 impostato **per istanza** (non per modello)</mark> nel loop di `updateUniformBuffer()`, confrontando con `gazedInstance`. Tutti gli altri campi qui sopra sono per-modello (arrivano da `materials.json`, che è chiavato per modello); questo no, perché di dieci porte identiche solo quella inquadrata deve brillare. Codifica due cose in uno scalare: la magnitudine arrotondata sceglie il colore (1 porta/oro, 2 pickup/viola, 3 candela/arancio), il segno negativo lo forza a rosso (interazione al momento non disponibile).
- `metallic` — forza `k = 0` (un conduttore non ha diffuso: gli elettroni liberi assorbono la luce che entra invece di ri-emetterla) e sostituisce l'ambient diffuso con `metalAmbient()`. Il flag porta l'intera definizione di "questo è un metallo" in un posto solo, così nessuna voce di `materials.json` può restarsi dietro un `k` diffuso per sbaglio.

## 2.4 `GlobalUniformBufferObject` — set 0, per-frame

Definito a main.cpp:107. La divisione rispetto al precedente è per **frequenza
di scrittura**: questo viene scritto una volta per frame, quello ~23 volte.

Layout std140, offset per offset:

- offset 0: `vec3 eyePos`, offset 12: `int lightCount`
- offset 16: `vec3 ambientUpper` (colore del cielo)
- offset 32: `vec3 ambientLower` (colore del terreno)
- offset 48: `vec3 ambientDir`, offset 60: `int debugFlags`
- offset 64: `float time`, offset 68: `float ambientWeight`
- offset 80: `LightData lights[MAX_LIGHTS]`, con `MAX_LIGHTS = 32`

`debugFlags`, `time` e `ambientWeight` viaggiano nel padding che precede
comunque l'array, che partirebbe a un multiplo di 16 in ogni caso.

L'array è a **dimensione fissa con un contatore separato** perché un uniform
block richiede una dimensione nota a tempo di compilazione: non esistono array a
lunghezza variabile. `lightCount` dice quanto dell'array è reale.

Nota per onestà: il commento in main.cpp:116 dice che l'array parte a 64.
Contando gli offset std140 parte a 80. Non è un bug — C++ e GLSL concordano
comunque, perché `LightData` ha `alignas(16)` da entrambi i lati — ma il numero
scritto nel commento è vecchio.

`LightData` (SceneLights.hpp:41) è **64 byte esatti**, con lo stesso trucco
scalare-dopo-vec3 ripetuto tre volte:

- `vec3 pos` + `float g` — posizione (solo point/spot) e distanza di riferimento del decadimento
- `vec3 dir` + `float beta` — direzione e esponente di decadimento (0 costante, 1 lineare, 2 quadratico)
- `vec3 color` + `float cosIn` — colore emesso e coseno del semiangolo interno del cono
- `float cosOut`, `int type`, `int shadowIndex`, + 4 byte di padding

`shadowIndex` = -1 significa "non proietta ombra"; altrimenti è un indice
nell'array di shadow map 2D (se la luce è direct o spot) oppure nell'array di
cubemap (se è point). Sta lì perché quei 4 byte erano già sprecati: `cosOut` +
`type` occupavano 8 byte di uno slot da 16 che l'array padda comunque.

**Riuso del set 0 da parte delle fiamme.** `Flame.vert`, `Flame.frag` e
`Spark.vert` bindano **lo stesso identico descriptor set** (`DSglobal`) come set
0, ma dichiarano il blocco fermandosi a `time` e omettendo `lights[]`. È lecito
perché gli offset std140 sono **posizionali**: uno shader può smettere di
dichiarare prima della fine, ma non può saltare né riordinare niente di quello
che precede un campo che legge. Alle fiamme serve solo `time`, e riusare il set
esistente evita di tenere in sync una seconda copia di `eyePos`, `lightCount`
eccetera.

## 2.5 `ShadowUniformBufferObject` — set 2, campionamento delle ombre

Un solo campo: `mat4 lightSpace[NUM_SHADOW_MAPS_2D]`, cioè la stessa
view-projection con cui ogni shadow pass 2D ha renderizzato la sua mappa. Serve
a `shadowFromMap2D()` in `CookTorrance.frag` per portare il frammento nello
spazio di quella mappa e confrontare la profondità.

Le torce **non hanno una matrice qui**: una cubemap si campiona per *direzione*,
non trasformando in clip space, quindi la loro matematica non lascia mai
`computeShadowMatrices()` / `populateCommandBuffer()`.

Il set 2 contiene anche i sampler:

- binding 0 — l'UBO qui sopra
- binding 1..2 — i `sampler2D` delle mappe 2D
- binding 3..34 — i 32 `samplerCube`

Sono **binding separati e non un array binding**. Motivo tecnico:
`Scene::init` calcola la dimensione del descriptor pool facendo
`texturesInPool += 1` per ogni *binding*, non per ogni *descrittore* che un array
binding richiederebbe. Un array binding sotto-dimensionerebbe silenziosamente il
pool. Da qui le catene `if(idx == 0) ... if(idx == 1) ...` in
`sampleShadowMap2D()` e `sampleShadowCube()`: è il prezzo di quella scelta.

Questo set passa attraverso la macchina per-istanza di `Scene` (a `P` vengono
dati tre layout), quindi **ogni istanza CookTorrance ne riceve una copia
identica e ridondante**. È uno spreco consapevole: a questo numero di istanze
costa poco, ed evita di scrivere un terzo modo di bindare un descriptor set
accanto ai due che già esistono.

## 2.6 `ShadowCubeUniformBufferObject` — set 1 del pass di cattura cubemap

- `mat4 lightViewProj[6]` (offset 0, 384 byte) — le sei view-projection, una per faccia
- `vec4 lightPos` (offset 384) — xyz usati, w è padding per tenere il blocco allineato a 16

Uno per slot cubemap (`DSshadowCube[NUM_SHADOW_CUBES]`), ri-mappato ogni frame
per **tutte** le torce, anche quelle statiche, per il motivo del §2.1.

La faccia corrente arriva invece come push constant
(`ShadowCubeFacePushConstant`), ed è sicuro perché è una proprietà di *dove* sta
il draw nel command buffer, non di quando gira.

## 2.7 `FlameUniformBufferObject` — set 1 di fiamme e scintille (112 byte)

Definito in `Flame.hpp:66`, letto identico da `Flame.vert` e `Spark.vert`:

- `mat4 mvpMat` (offset 0) — la base del billboard moltiplicata per ViewPrj, costruita dalla CPU
- `float seed` (64) — sfasamento per-torcia, così due torce non ondeggiano all'unisono
- `float intensity` (68) — inviluppo di **luminosità**, ~0.30..1.40
- `vec2 lean` (72) — inclinazione dovuta al movimento della mano
- `float heightScale` (80) — inviluppo di **altezza**, ~0.78..1.09
- `float glareBoost` (84) — enfasi "sto fissando questa fiamma"
- `vec3 color` (96, con `alignas(16)`) — tinta bersaglio

L'offset 96 e non 88 è std140 in azione: dopo `glareBoost` saremmo a 88, ma un
`vec3` vuole un offset multiplo di 16, quindi si salta a 96.

Le scintille riusano lo stesso descriptor set della fiamma: a una scintilla
servono solo la mvp e il seed della torcia, non c'è motivo di tenere in sync una
seconda copia.

## 2.8 `PostUniformBufferObject` — set 0 dei quattro pass di post-processing

Un solo blocco condiviso da bright pass, i due blur e il composite. Ogni pass ha
la sua copia e riempie i campi che gli servono, lasciando gli altri semplicemente
non letti — più economico che mantenere quattro blocchi quasi identici.

- `vec2 texelSize` — 1/larghezza, 1/altezza della texture **sorgente**
- `vec2 blurDir` — (1,0) o (0,1), letto solo da `BloomBlur.frag`
- `float threshold`, `float knee` — bright pass
- `float bloomIntensity`, `float exposure` — composite
- `int debugFlags` — composite (`LIGHT_DEBUG_NO_TONEMAP`)
- `float time`
- `float escapeFlash` — 0..1, il bianco della fuga

Perché `escapeFlash` è un termine a sé e non semplicemente "più esposizione":
vedi §7.4, è una domanda che si presta bene a essere fatta.

## 2.9 `ExitGlowUniformBufferObject` — set 0 del pass ExitGlow

`mat4 mvpMat`, `vec3 color`, `float intensity`, `float time`, `float softness`.
Nessun global UBO bindato qui: niente in questo effetto è illuminato, quindi
`eyePos` e l'array di luci resterebbero non letti, e `time` — l'unica cosa che
sarebbe servita — viaggia dentro questo blocco.

## 2.10 Riepilogo dei descriptor set layout

- `DSLlocal` — binding 0: UBO per-istanza, visibilità `ALL_GRAPHICS` (non solo `VERTEX_BIT`, perché questo buffer porta anche il materiale, che legge il fragment); binding 1: `COMBINED_IMAGE_SAMPLER` (l'albedo).
- `DSLglobal` — binding 0: UBO globale, `ALL_GRAPHICS`.
- `DSLshadowSample` — binding 0 UBO + 2 `sampler2D` + 32 `samplerCube`, tutti `FRAGMENT_BIT`. Costruito con un loop e non a mano, così il conteggio vive in un posto solo.
- `DSLshadowCubeCapture` — binding 0: UBO, visibile a `VERTEX | FRAGMENT` (il vertex legge le matrici, il fragment la posizione della luce).
- `DSLpost1` — UBO + 1 texture. `DSLpost2` — UBO + 2 texture (il composite deve mescolare scena e bloom).

La pipeline principale è
`P.init(this, &VD, "PosNormUV.vert.spv", "CookTorrance.frag.spv", {&DSLglobal, &DSLlocal, &DSLshadowSample})`:
l'ordine del vector è l'ordine dei set, quindi set 0 / 1 / 2.

Curiosità sull'API del framework che vale la pena sapere se te lo chiedono:
nella dichiarazione di un binding sampler, il secondo numero **non è una
dimensione in byte** ma l'indice dentro il vector di `VkDescriptorImageInfo`
passato a `DescriptorSet::init`. Starter.hpp riusa il campo `linkSize` per due
scopi diversi a seconda del tipo di binding.

> **Se il prof chiede**
>
> *"Perché tre descriptor set e non uno?"* — Per frequenza di aggiornamento: set
> 0 cambia una volta per frame, set 1 una volta per oggetto, set 2 mai (dipende
> dalle luci che proiettano ombra). Bindare costa, e mettere tutto insieme
> costringerebbe a ri-bindare tutto per cambiare una sola cosa.
>
> *"Perché non usi push constant per le matrici, che sono più veloci?"* — Perché
> il command buffer è registrato una volta per immagine della swapchain e poi
> riprodotto identico. Una push constant sarebbe congelata a quel valore. Le
> uso infatti nei due casi in cui il valore è davvero costante o dipende dalla
> posizione nel buffer: la view-projection del sole e l'indice della faccia della
> cubemap.
>
> *"Cosa succede se sbagli l'allineamento std140?"* — Lo shader legge i campi a
> offset sbagliati. Non c'è nessun errore, il risultato è semplicemente errato:
> materiali strani, matrici a caso. Si previene con `alignas(16)` sui vec3/mat4
> lato C++.
>
> *"Perché `nMat` è una mat4?"* — Perché std140 padda ogni colonna di una mat3 a
> 16 byte, e le regole sono facili da sbagliare. Una mat4 costa 16 byte in più
> ed è priva di ambiguità.
>
> *"Perché l'array di luci ha dimensione fissa?"* — Un uniform block deve avere
> dimensione nota a compile time. Da qui `MAX_LIGHTS` più un `lightCount` che
> dice quanto dell'array è reale.

---

# PARTE 3 — Gli shader, uno per uno

## 3.1 `PosNormUV.vert` — il vertex shader della scena

Fa due cose:

```glsl
gl_Position = ubo.mvpMat * vec4(inPosition, 1.0);
fragPos  = (ubo.mMat * vec4(inPosition, 1.0)).xyz;
fragNorm = mat3(ubo.nMat) * inNormal;
fragUV   = inUV;
```

Perché `nMat` e non `mMat` per la normale: se la matrice di mondo ha una **scala
non uniforme**, trasformare la normale con essa la fa smettere di essere
perpendicolare alla superficie. Esempio concreto nella nostra scena: la strada è
scalata `[1, 4, 1]`. La matrice giusta è l'inversa trasposta.

La normale **non viene normalizzata qui**: tanto l'interpolazione della
rasterizzazione la accorcia comunque (la media di due versori non è un versore),
quindi deve rinormalizzare il fragment shader. Farlo due volte sarebbe lavoro
sprecato in uno stadio che gira migliaia di volte.

Il blocco `UniformBufferObject` è dichiarato per intero anche se qui si usano
solo tre matrici: il blocco deve essere identico nei due stage, perché è **un
solo buffer a un solo binding, condiviso**.

## 3.2 `CookTorrance.frag` — il cuore del progetto

936 righe. Fa, in ordine:

1. Normale: `normalize(fragNorm)`, oppure normale di faccia se `flatNormals`.
2. Albedo dalla texture. Nessuna conversione sRGB a mano: la fa l'hardware (§0.8).
3. Le viste di debug con uscita anticipata (normali come colore, unlit, heatmap).
4. Il grime procedurale per i metalli d'interno (§3.3).
5. Il loop sulle luci: `Lo += radiance * BRDF(...) * shadowFactor(...)`.
6. Il termine ambient (emisferico, oppure `metalAmbient()` per i metalli).
7. Il blend: `color = Lo * (1 - aw) + ambient * aw`.
8. Il focus glow (§3.4).
9. Scrittura **lineare e non clampata** nell'attachment RGBA16F.

### L'equazione del rendering, in pratica

Per ogni luce si calcola quanta radianza arriva al punto, e quanta di quella
rimbalza verso l'occhio. Il secondo fattore è la **BRDF** (Bidirectional
Reflectance Distribution Function).

Usiamo **Cook-Torrance con GGX** (E06 s.38-39):

```
fr = clamp(N·L) * ( k*mD + (1-k) * mS * D*F*G / (4*clamp(N·L)*clamp(N·V)) )
```

L'idea fisica: una superficie non è liscia, è fatta di microfacce ognuna delle
quali è uno specchietto perfetto. La riflessione speculare che vedi è dovuta a
quelle microfacce orientate esattamente a metà tra la direzione della luce e
quella dell'occhio (la **half vector** `h`). I tre termini rispondono a tre
domande:

- **D — `distributionGGX(N, h, roughness)`**. "Quante microfacce sono orientate lungo `h`?" Una distribuzione statistica: rugosità bassa = quasi tutte allineate alla normale media = riflesso stretto e netto. Rugosità alta = orientazioni sparse = riflesso largo e morbido. GGX ha code lunghe, il che è ciò che dà ai riflessi reali il loro alone.
- **G — `geometricTerm(N, h, L, V)`**. "Quante di quelle microfacce sono nascoste da altre microfacce?" A incidenza radente una microfaccia fa ombra alla vicina, oppure la nasconde alla vista. Senza questo termine le superfici ruvide diventano innaturalmente brillanti ai bordi. Usiamo la forma senza parametri (E06 s.47).
- **F — `fresnelSchlick(V, h, F0)`**. "Che frazione viene riflessa invece che trasmessa?" Ogni superficie diventa uno specchio se la guardi abbastanza di striscio: pensa a un tavolo di legno visto quasi di lato. `F0` è la riflettanza guardando in faccia; l'approssimazione di Schlick è `F0 + (1-F0)*(1-V·H)^5`. Il 5 è un fit empirico, non deriva da niente.

Il denominatore `4*(N·L)*(N·V)` è la normalizzazione da area di microfaccia ad
area di superficie, non un fattore di aggiustamento arbitrario.

Diffuso e speculare sono **interpolati** con `k`, non sommati: sommandoli a piena
forza la superficie restituirebbe più luce di quanta ne ha ricevuta, che è
fisicamente impossibile.

### Perché il tone map non è qui

C'era, ed è stato spostato in `Composite.frag`. Il motivo, che vale la pena
saper dire in una frase: il tone map schiaccia tutto dentro [0,1], e il bloom
funziona **cercando i pixel sopra 1**. Comprimere prima significa buttare via
l'unica informazione che il bright pass sta cercando, e la fiamma resterebbe
senza alone.

### L'ottimizzazione per-pixel del loop luci

```glsl
if(max(radiance.r, max(radiance.g, radiance.b)) < LIGHT_ATTEN_EPS) continue;
```

con `LIGHT_ATTEN_EPS = 1e-3`. Sotto quella soglia, la radianza di quella luce su
questo frammento è più scura di quanto l'immagine finale possa mostrare — meno
di 1 LSB su un display a 8 bit, dopo esposizione e curva sRGB. Il `continue`
salta il BRDF (che è pieno di `pow()`) e soprattutto la **fetch dipendente**
sulla shadow map, che è la cosa cara.

Serve perché le point light sono cullate dalla CPU solo per distanza dalla
**camera**, non dal frammento che si sta ombreggiando: una torcia che ha passato
quel cull può comunque essere quasi a zero su un frammento dall'altro lato di
una sala grande.

### Il termine ambient

L'**ambient emisferico** (E07 s.47-54) è un modello economico della luce
indiretta: invece di simulare i rimbalzi, si assume che dall'alto arrivi il
colore del cielo e dal basso quello riflesso dal terreno, e si mescolano in base
a come è orientata la superficie.

```glsl
w = (dot(dir, ambientDir) + 1.0) / 2.0;   // dot è -1..1, w è 0..1
return mix(ambientLower, ambientUpper, w);
```

`ambientShare()` restituisce `ubo.ambientWeight` se ≥ 0, altrimenti
`gubo.ambientWeight`: un peso per-modello che sovrascrive quello di scena, così
un corridoio chiuso e un cortile aperto possono avere valori diversi nello stesso
frame. Il default è il valore *da interno*, perché il gioco si svolge dentro il
dungeon.

Il punto importante — e una probabile domanda — è che il risultato è un **blend,
non una somma**:

```glsl
color = Lo * (1.0 - aw) + ambient * aw;
```

Sommando, come faceva la versione precedente, l'ambient diventava un pavimento di
luminosità sotto ogni pixel della scena: non ha alcun termine di visibilità,
quindi una stanza sigillata raccoglieva la stessa luce indiretta del cortile
aperto e nessun soffitto poteva fermarla. Col blend, l'ambient non può mai
contribuire più di `aw` al frame, e il totale non può mai superare quello che le
luci dirette da sole avrebbero dato.

Va detto onestamente: **non è occlusione**. È una stima autorata a mano di quanto
una superficie sia racchiusa, che è quello che fa E17 e quello che gli asset
permettono — le mesh MGCG spediscono solo l'albedo, quindi non c'è nessuna mappa
di ambient occlusion da campionare. La soluzione onesta sarebbe bakerne una.

Nota importante: l'ambient **non viene moltiplicato per lo shadow factor**. Lo
shadow mapping blocca solo il contributo diretto di una luce, mai il rimbalzo
indiretto: altrimenti un'ombra si leggerebbe come un buco nel nero assoluto
invece che come la zona fiocamente illuminata che è nella realtà.

### `metalAmbient()` — l'ambient per i metalli

Un metallo non ha diffuso, quindi quasi tutto quello che vedi guardandolo è un
**riflesso di ciò che gli sta intorno**. L'ambient diffuso non può esprimerlo:
darebbe alla superficie un colore scelto in base a *dove è rivolta*, per un
albedo, che è l'unica cosa che un metallo non fa. Lasciata così, la catena
usciva quasi nera ovunque non arrivasse una torcia (le sue UV campionano la
banda di ferro battuto della texture della porta, albedo ~0.03 lineare, e `mD`
azzerava il termine) e il lucchetto usciva come plastica arancione dipinta.

La scena non ha environment map, quindi **l'emisfero fa da ambiente**: si
campiona `hemisphereColor()` lungo la direzione riflessa. Con due accorgimenti:

- un metallo ruvido riflette una stanza *sfocata*, e con un emisfero a due colori e niente mip chain non c'è nulla da sfocare. Il sostituto è far scivolare la direzione di campionamento da `reflect(-V, N)` (specchio) verso `N` (quello che userebbe una superficie diffusa) man mano che la rugosità cresce: i due estremi sono esattamente i due estremi che il modello vero interpola.
- Schlick di nuovo, ma su `N·V` (non c'è half vector, perché la "luce" è tutto l'emisfero). Il tetto è `max(1 - roughness, F0)` invece di 1.0: un metallo ruvido non diventa uno specchio perfetto all'orizzonte, e lasciarlo arrivare a 1.0 metteva un bordo netto e brillante esattamente sui pixel che disegnano il contorno di un tubo.

È la forma più economica e onesta dell'ambient speculare "split-sum" (l'emisfero
di E07 al posto di una cube map prefiltrata); quella vera richiederebbe un pass
di cattura che il progetto non ha.

## 3.3 Il grime procedurale

Per i metalli d'interno (catene, lucchetto, chiave). Quei tre oggetti sono
sottoterra da abbastanza tempo da essersi ossidati, e niente di tutto ciò è
nell'albedo piatta della pack MGCG. Senza un secondo set di UV e senza una dirt
map da campionare, lo sporco viene **generato dalla posizione di mondo**: value
noise 3D su `fragPos`, tre ottave (6 / 18 / 50 per unità), poi
`smoothstep(0.35, 0.80, g)` così il metallo pulito resta genuinamente pulito e
le chiazze sporche vanno quasi a fondo, invece di un velo grigio uniforme.

Usando la posizione di mondo, due anelli vicini della stessa catena escono
invecchiati in modo diverso invece che identico.

Effetto: rugosità su (lo sporco disperde ciò che il metallo nudo rimanderebbe
netto), tinta del riflesso giù (una superficie patinata riflette meno). L'albedo
non si tocca: un metallo ha `k = 0` e quindi non ha diffuso da sporcare.

Ottone contro acciaio non ha un flag: si legge `mS.b / mS.r` (~0.46 per l'ottone,
~1.04 per l'acciaio), perché il colore speculare è già il segno distintivo, e si
scala e si comprime lo sporco di conseguenza.

## 3.4 Il focus glow

L'aura sull'oggetto che il mirino sta inquadrando. Un termine **Fresnel/rim**
(`pow(1 - N·V, 2.2)`) lo concentra sulla silhouette invece di spalmarlo su tutta
la superficie, come un campo che aderisce ai bordi. `flow` è un seno su
`dot(fragPos, ...)` più il tempo, così l'onda **viaggia sulla superficie**
dell'oggetto invece di far pulsare tutto insieme.

C'è anche un contorno quasi nero, con una potenza molto più aggressiva
(`pow(1 - N·V, 12.0)`) così compare solo negli ultimi gradi, applicato **prima**
dell'oro. Su una superficie riflettente come la chiave, un highlight brillante
può altrimenti passare esattamente dove sta la banda dorata e fondere le due cose
in una sbavatura oro-su-oro; mettere qualcosa di più spento immediatamente sotto
è ciò che stacca l'aura dal modello, come un contorno stacca un adesivo dallo
sfondo.

Il dettaglio più interessante da saper spiegare: il tone map in `Composite.frag`
divide ogni pixel per **la luminanza di quello stesso pixel**. Un glow additivo
semplice verrebbe quindi schiacciato dove la superficie sotto è già luminosa, e
resterebbe pieno dove è buia — esattamente il contrario di "sempre visibile".
Moltiplicare l'aggiunta per `(1 + luminanza pre-glow)` cancella quella divisione
al primo ordine: l'algebra è `(c + k(Y+1)) / (Y + k(Y+1) + 1) → k/(1+k)` al
crescere di `Y`, cioè una costante invece di qualcosa che tende a zero.

## 3.5 `Shadow.vert` / `Shadow.frag` — le shadow map 2D

`Shadow.vert` è tre righe: `gl_Position = pc.lightViewProj * ubo.mMat * pos`.

Due cose da saper spiegare:

- riusa il **descriptor set per-istanza della pipeline principale**, dichiarato a set 0 qui perché questa pipeline non ha il `DSLglobal` davanti. È l'unico punto del renderer in cui un descriptor set è condiviso fra due pipeline diverse, ed è ciò che fa sì che un occlusore in movimento proietti un'ombra che lo segue.
- la view-projection della luce è una **push constant**, il che è sicuro solo perché il sole non si muove mai (§2.1).

`Shadow.frag` è vuoto. Il render pass è `AT_DEPTH_ONLY`, non ha alcun color
attachment: la profondità la scrive lo stadio fixed-function da `gl_Position.z`,
non c'è nessun valore per pixel che un fragment shader debba calcolare. Lo stage
esiste solo perché `Pipeline::init` linka sempre vertex + fragment.

## 3.6 `ShadowCube.vert` / `ShadowCube.frag` — la cattura delle cubemap

`ShadowCube.vert` legge lo stesso `ubo` per-istanza (a set 0), più `cubeData` (a
set 1) con le sei matrici e la posizione della luce, più la push constant
`face`. A differenza di `Shadow.vert` deve passare al fragment anche la posizione
di mondo del vertice.

`ShadowCube.frag` scrive `length(worldPos - lightPos)` in un attachment
`R32_SFLOAT`: la **distanza lineare in unità di mondo**, non la profondità
proiettiva del depth buffer.

Il perché è una domanda che il prof potrebbe benissimo fare.
`CookTorrance.frag` campiona questa cubemap **per direzione**, con un
`samplerCube`, dando come coordinata `fragPos - lightPos`. Quella lookup non ha
alcuna nozione di *quale faccia* ha risposto. Quindi il valore contro cui si
confronta deve significare la stessa cosa su ogni faccia: vero per una distanza
euclidea, falso per una profondità prospettica, che si deforma diversamente al
centro di una faccia rispetto al bordo, e diversamente ancora sulla faccia
adiacente che condivide quel bordo.

Il clear value dell'attachment è `TORCH_SHADOW_FAR_CONST = 60.0`: senza nessun
occlusore disegnato, la "distanza" deve leggersi come infinito, cioè illuminato.

## 3.7 `Flame.vert` / `Flame.frag` — il corpo della fiamma

`Flame.vert` è deliberatamente semplice: piazza **tre card piatte** in spazio
locale e le fa deformare dagli inviluppi. `x` copre la mezza larghezza, `y` va da
0 (stoppino) a 1 (punta) — non da -1 a 1, perché una fiamma è ancorata alla base
e cresce verso l'alto, e avere y=0 sullo stoppino è ciò che permette agli shader
di scalare l'altezza e applicare l'inclinazione senza prima dover disfare un quad
centrato.

Le tre card non sono rettangoli identici sovrapposti: hanno larghezze
leggermente diverse (1.12 → 0.88 dalla più lontana alla più vicina) e sono
separate lungo z di 0.10, così non sono complanari e la camera che gira attorno
alla torcia legge parallasse invece che una card piatta.

Il `lean` è pesato `h*h` e non `h`: bloccato alla base, massimo in punta, che è
dove sta l'oscillazione di una fiamma vera.

`Flame.frag` è dove sta tutto l'aspetto. È **unlit**: il fuoco emette, non
riflette la luce della scena, quindi non c'è un lato illuminato e uno in ombra da
calcolare. La geometria sono sei triangoli; tutta la forma viene dal campo
procedurale valutato per pixel e poi ritagliato con `discard`.

I passi, in ordine:

1. **Spina dorsale ondeggiante.** La mediana stessa oscilla: la `x` di campionamento viene spostata da rumore lento avvettato lungo l'asse della fiamma, zero allo stoppino (una fiamma è ancorata al suo combustibile) e massimo in punta (`y*y`). Tutto ciò che viene dopo lavora in questa x relativa alla spina, quindi **l'intero campo si arriccia, silhouette e struttura interna insieme**, invece di far scorrere la texture dentro un ritaglio fermo. È la differenza singola più grande fra "pulsa sul posto" e "brucia": senza, il contorno della fiamma non si sposta mai.
2. **Maschera di profilo.** La semi-larghezza `w(y)` è strozzata allo stoppino, massima poco sopra, e si assottiglia a punta. `wFall = pow(1-y, 0.65)` è una **potenza**, non uno smoothstep, ed è tutta la differenza fra una fiamma e un aquilone: una rampa lineare dà fianchi dritti che incontrano il punto più largo ad angolo, cioè un rombo; la potenza scende piano all'inizio e sempre più in fretta, quindi i fianchi si inarcano in fuori in basso e si chiudono a punta in alto. Poi `shape = 1 - r²` (parabolico e non lineare attraverso la larghezza) tiene pieno il nucleo e cade più in fretta verso il bordo, così il corpo si legge tondo invece che come un cuneo con una piega brillante al centro.
3. **Avvezione.** Si **sottrae** il tempo dalla y di campionamento. Sottrarre e non sommare è ciò che fa *salire* il pattern invece di farlo pulsare sul posto: un punto fisso sullo schermo vede, col passare del tempo, il valore di rumore che prima stava più in basso — cioè combustibile che sale visibilmente.
4. **Domain warp.** L'FBM non viene campionato direttamente: viene campionato in un punto **spostato da un secondo FBM** più lento. Questa è la differenza vera fra "lingue che leccano" e un gradiente frizzante — senza, la luminosità varia dolcemente sul posto; deformando il *punto di campionamento*, le zone brillanti stesse si arricciano e viaggiano di lato mentre salgono. La coordinata del warp ha anche un termine di tempo laterale, così il campo si deforma oltre che traslare: una lingua cambia forma mentre sale, invece che essere sempre lo stesso ricciolo congelato che risale. E il morso del warp cresce con l'altezza: quasi laminare allo stoppino, dove una fiamma vera è un cono liscio, pienamente turbolento in punta.
5. **Carve.** Quanto il rumore può mordere la silhouette, `mix(0.35, 1.30, y)`. Vicino allo stoppino la fiamma è densa e stabile, quindi il rumore la perturba appena; verso la punta è sottile e turbolenta, e lì il rumore può spingere il campo abbastanza in negativo da **staccare pezzi dalla silhouette**. È così che nascono i filamenti che si distaccano e volano via.
6. **Bolle.** Sacche di gas più caldo che si formano in basso, salgono con l'avvezione e muoiono verso la punta. Un campo a bassa frequenza **sogliato** con smoothstep, non altre ottave di FBM: la soglia trasforma il rumore in blob distinti con vuoti in mezzo, che è cosa sono le bolle — altre ottave renderebbero solo più affollata la texture esistente. Scorrono ~35% più veloci del corpo, quindi le sacche **sorpassano visibilmente** la texture su cui viaggiano, e si leggono come volumi che galleggiano invece che luminosità dipinta.
7. **Rampa di temperatura**, agganciata a `heat` (il campo di fuoco intagliato dal rumore) e **non** all'altezza sulla mesh: i pixel più caldi e bianchi sono ovunque il rumore dica che il nucleo è adesso, e quel punto deriva e lecca verso l'alto invece di stare sempre alla stessa altezza. Quattro stop da rosso scuro a bianco caldo.
8. **Ricolorazione.** Ogni torcia ha un `color`. Invece di sostituire gli stop (che o non riprodurrebbe l'aspetto attuale o richiederebbe un caso speciale fragile per il default), **ogni stop viene ruotato in tinta** verso `color` dello stesso delta. Così si conserva la progressione di valore e saturazione (scuro → bianco caldo) che è ciò che fa leggere il gradiente come fuoco, e con il colore di default (l'arancione) il delta è zero e la palette è esattamente quella di prima.
9. **HDR boost** fino a 6× sul nucleo, così il bloom a valle ha energia vera da trovare, lasciando il bordo vicino a 1 così non sbianca anche quello perdendo il rosso.
10. **Shimmer**, un flicker per-pixel veloce che **varia lungo la fiamma**: altezze diverse guizzano in momenti diversi, che si legge come combustione invece che come una manopola della luminosità mossa avanti e indietro. Il flicker lento sta in CPU perché la point light deve seguire lo stesso segnale, e una luce non può comunque sfarfallare per-pixel.

Il `discard` sotto alpha 0.04: vedi §0.7, è obbligatorio, non è
un'ottimizzazione.

## 3.8 `Spark.vert` / `Spark.frag` — le scintille

Interamente procedurali: **nessun particle system lato CPU** e nessun dato
uniforme per particella oltre a ciò che è già nella mesh. Ogni scintilla ha un
`inSeed` fisso, uguale su tutti e quattro i vertici del suo quad (così il quad si
muove come una particella sola invece di vedere gli angoli separarsi). Nascita,
salita, deriva, allungamento, morte sono tutte funzioni di `gubo.time` e di quel
seed, in loop con `fract()`.

- Periodo 1.6..3.2 s, sfasato **sia** dal seed della scintilla **sia** da quello della torcia, così scintille di torce diverse non tornano allo spawn all'unisono.
- Nascono sulla **corona** (y 0.42..0.62), non sullo stoppino: una scintilla è un fiocco che la punta scaglia via, non combustibile che sale dalla base come il corpo della fiamma.
- La deriva laterale cresce con `t²`, non con `t`: una scintilla vaga appena dopo essere stata espulsa e si incurva sempre di più mentre si raffredda e rallenta, invece di derivare a velocità costante per tutta la vita.
- Sopra c'è un **wobble**, una passeggiata di rumore campionata **lungo il tempo**, così è movimento e non una curvatura statica. Anche la sua ampiezza cresce con l'età: una scintilla giovane è ancora portata dalla corrente ascensionale, una vecchia è in balia dei vortici.
- La corona segue `heightScale`: quando la fiamma cala, le scintille nascono e volano più in basso invece di spuntare dal vuoto sopra una fiamma rimpicciolita.
- Lo **"spawn rate" senza mesh dinamica**: il numero di scintille è fisso, quindi il rate viene simulato dando a ogni scintilla la sua **soglia** di intensità (0.45..1.15) contro un inviluppo che varia 0.30..1.40. Il ~40% con le soglie più basse brucia sempre, il resto arriva a raffiche quando la fiamma divampa o torna di colpo dopo un guttering. Siccome l'inviluppo è smorzato in CPU, la banda dello smoothstep diventa una dissolvenza di 0.1-0.3 s invece di un pop.
- Una scintilla spenta collassa in un quad degenere, quindi non costa alcun lavoro di fragment mentre aspetta.
- Allungate 2.5..4× lungo la direzione di moto, così si leggono come strisce e non come punti. La direzione è la derivata della stessa curva del centro: la salita è quasi costante e la deriva cresce, quindi una scintilla giovane va quasi dritta in su e si inclina invecchiando.
- Il fragment shader usa `length(quv)` sul corner **non allungato**. Siccome l'allungamento è una mappa affine (lineare), interpolare il corner grezzo sul quad già allungato combacia perfettamente con la forma fisica: un falloff circolare su `quv` si legge automaticamente come un'ellisse stirata lungo la striscia, senza dover passare anche il fattore di allungamento.
- HDR boost da 12× alla nascita a 3× alla morte: una scintilla è una vera sorgente di bloom solo appena espulsa, non per tutta la sua breve vita.

## 3.9 `BloomBright.frag` — la soglia

Legge il target HDR a piena risoluzione e scrive a **un quarto** di risoluzione
per lato.

**Quattro tap diagonali a mezzo texel.** Il filtraggio bilineare media ciascun tap
su un blocco 2×2 di texel sorgente, quindi i quattro campioni coprono fra loro
tutta l'area sorgente che mappa su questo texel di uscita: un box downsample al
prezzo di 4 fetch invece dei 16 ingenui. Fare soglia e downsample nello stesso
pass invece che in due dimezza il traffico di texture gratis.

**Karis average.** Ogni tap è pesato `1/(1+luma)` invece di essere mediato
uniformemente. Senza, una singola scintilla molto più brillante dei vicini
domina la media e fa lampeggiare il quad downsamplato al ritmo delle singole
scintille invece che a quello della luminosità complessiva della fiamma.

**Soglia con soft knee** (Karis, SIGGRAPH 2014). Un taglio netto alla soglia fa
sfarfallare la maschera pixel per pixel man mano che le scintille attraversano la
linea da un frame all'altro; la rampa quadratica dentro
`[soglia-knee, soglia+knee]` fa entrare un pixel gradualmente man mano che si
illumina. Il colore originale viene poi **riscalato dalla frazione di luminosità
che sopravvive**, invece di restituire un grigio piatto: così tinta e saturazione
restano intatte sulle parti che fanno bloom.

**Il clamp ai bordi.** Tutti i tap sono clampati a mezzo texel dal bordo, e non
è pignoleria. `Starter.hpp` costruisce il sampler di ogni framebuffer attachment
con il suo address mode di default, che è `VK_SAMPLER_ADDRESS_MODE_REPEAT`.
Senza il clamp, i tap che escono da un bordo **avvolgono** su quello opposto, e
qualcosa di sovraesposto in cima al frame stampa una banda brillante in fondo
allo schermo, e viceversa. Il clamp a mano equivale a `CLAMP_TO_EDGE` e tiene la
correzione dentro la catena del bloom, invece di cambiare un default di sampler
su cui ogni texture del livello conta per il tiling delle UV.

**La soglia scelta**, `BLOOM_THRESHOLD = 1.55`, non 1.0. Un muro assolato con
albedo chiara atterra poco sopra 1.0 una volta sommati sole e ambient, quindi a
1.0 ogni superficie brillante del castello prendeva un alone e il frame intero
sembrava annebbiato. La fiamma scrive fino a ~6 e le scintille di più: c'è un
corridoio largo in cui sedersi, e ci entra solo ciò che è genuinamente emissivo.

## 3.10 `BloomBlur.frag` — la sfocatura

Una gaussiana **separabile**: una gaussiana 2D è matematicamente scomponibile in
due passate 1D, con lo stesso risultato ma `2N` tap invece di `N²`. Questa
shader è la passata 1D, e quale asse percorre lo decide interamente
`post.blurDir`, che è il motivo per cui una sola shader serve per due pass.

Nove fetch indipendenti con pesi gaussiani discreti di raggio 4, normalizzati a
somma 1 (così la sfocatura non può cambiare la luminosità totale dell'immagine,
solo spargerla). **Non** il trucco a 5 fetch con tap bilineari (accoppiare i
pesi e campionare fra due centri di texel per avere due tap al prezzo di uno):
quel trucco richiede di combinare a mano i pesi di ogni coppia e risolverne il
punto di campionamento per ogni kernel, e sbagliare quell'aritmetica sbilancia
la sfocatura silenziosamente invece di dare errore. A un quarto di risoluzione i
quattro fetch in più costano poco.

Stesso clamp ai bordi di `BloomBright.frag`, stesso motivo.

E il quarto di risoluzione non è solo risparmio: un kernel a larghezza fissa su
un'immagine a un quarto **copre quattro volte più immagine finale**, ed è così
che si ottiene un alone largo e morbido da un kernel economico a 9 tap invece di
uno stretto.

## 3.11 `Composite.frag` — l'ultimo pass

```glsl
color = scene + bloom * bloomIntensity;
color *= exposure;
if(!(debugFlags & LIGHT_DEBUG_NO_TONEMAP)) color = toneMap(color);
color = mix(color, vec3(1.0), clamp(escapeFlash, 0.0, 1.0));
```

`toneMap(c) = c / (Y + 1.0)` con `Y` la luminanza Rec.709. Divisione per
luminanza e non per canale (§0.8).

**L'ordine conta.** `escapeFlash` è applicato **dopo** il tone map. Prima
sarebbe solo altra esposizione, e la curva `c/(Y+1)` — che tende al bianco senza
mai arrivarci — se lo mangerebbe: il frame diventerebbe pallido e si fermerebbe
lì, con le parti brillanti ancora chiaramente più brillanti. Dopo la curva non
c'è più niente che lo comprima, quindi `escapeFlash = 1` è genuinamente bianco.

Il fetch del bloom è clampato a mezzo **texel di bloom** dal bordo. Anche senza
alcun offset aggiunto, l'impronta bilineare dell'upsample esce dall'immagine
nell'ultimo mezzo texel — e a un quarto di risoluzione quel mezzo texel sono due
pixel a piena risoluzione: una linea brillante sottile in cima o in fondo allo
schermo ogni volta che qualcosa di sovraesposto sta all'altro capo del frame.

L'uscita è **lineare**, non gamma-codificata: la swapchain è `B8G8R8A8_SRGB` e
la codifica la fa l'hardware in scrittura (§0.8).

## 3.12 `Post.vert`

Condiviso da tutti e quattro i pass di post: tutti disegnano esattamente la
stessa cosa, un quad a schermo intero. `inPos` è già in NDC (gli angoli vanno da
(-1,-1) a (1,1)), quindi va dritto in `gl_Position` senza matrici — non c'è
camera né modello da cui trasformare. `uv = inPos * 0.5 + 0.5`.

Non ha nessun uniform block, deliberatamente: ogni pass legge i propri parametri
nello stadio fragment, perché niente di quella roba influenza dove finisce un
vertice. I vertici di un quad di post-processing non si muovono mai.

## 3.13 `ExitGlow.vert` / `ExitGlow.frag`

La luce del giorno dietro la porta d'uscita. **Non è un billboard**: è un piano
fisso in piedi sul terreno appena fuori dal varco, quindi ha un orientamento
reale nel mondo e `main.cpp` lo cuoce direttamente dentro `mvpMat`. Il vertex
shader passa il corner così com'è, non normalizzato: il fragment vuole la
distanza dal centro **del quad**, e interpolare il parametro del corner è
esattamente quella.

Il fragment shader fa tre scelte che vale la pena saper difendere:

**Distanza rettangolare, non radiale.** `m = max(|x|, |y|)`, così le curve di
livello sono il contorno del quad e non un'ellisse inscritta. Conta più di
quanto sembri: questi quad si vedono attraverso un arco da ogni angolo in cui un
giocatore può stare, e l'angolo peggiore proietta l'apertura su un **vertice**
del quad — che sotto `length()` è il punto più lontano che esista, quindi un
fade radiale ci arrivava mentre il centro piatto era ancora sprecato. Misurare
per asse mette a lavoro tutto il rettangolo e lascia al fade solo un bordo.

**Un plateau, non un falloff.** Il campo è piatto e completamente sovraesposto su
tutto il corpo del quad e cede solo nell'ultimo `softness`. Una versione
precedente sfumava dal centro verso l'esterno, e attraverso l'arco si leggeva
come una macchia calda che galleggiava su uno sfondo visibilmente più scuro —
cioè diceva al giocatore che là fuori c'è una superficie, l'unica cosa che questo
effetto non deve mai fare. Niente oltre il varco può avere una forma, quindi non
ce l'ha neanche la luce. Il fade sul bordo resta perché i quad sopravanzano ciò
che coprono su ogni lato: serve a non far vedere uno spigolo se un angolo di
vista dovesse mai coglierne uno, e dentro l'apertura non compare mai.

**Alpha che collassa molto più in fretta della luminosità.** La luce del giorno
deve **nascondere** ciò che c'è dietro: qualunque traslucenza e il piano del
terreno si vede attraverso il bagliore, che è esattamente ciò che il plateau
esiste per impedire.

Più il `discard` se `intensity <= 0.001` (porta chiusa): non si può saltare la
draw call, il command buffer è registrato una volta sola (§0.6).

E la scelta architetturale: **il bagliore non è geometria**. Non c'è nessun
billboard di glow, nessun alone modellato, nessun secondo pass. Questa shader
disegna un'ellisse molto brillante — ben oltre 1.55, la soglia del bright pass —
e la catena di bloom la trasforma in un abbaglio che sborda sul telaio della
porta. Stessa identica logica delle fiamme.

## 3.14 `SpectralDepth.frag` — il depth prepass dei fantasmi

I fantasmi sono alpha-blended (`Pspectral`, `setTransparency(true)`,
main.cpp:2709), quindi tutto ciò che sta dietro a una loro superficie traspare
attraverso — **compresi loro stessi**. I piedini stanno dentro la veste, e li si
vedeva brillare attraverso il corpo: il rim di Fresnel è la cosa più luminosa
che quella shader produce, quindi è anche quella che sopravvive meglio a una
fusione al 46%.

**Perché il depth test da solo non bastava.** Il mesh esce in ordine di indice,
e il piedino capita prima della veste. Su un pixel dove il muro è a z 0.90, il
piedino a 0.55 e la veste a 0.50:

- piedino, `0.55 < 0.90` → passa, si fonde col muro, e **scrive 0.55**
- veste, `0.50 < 0.55` → passa, si fonde **sopra** il piedino, scrive 0.50

Il pixel finale è `0.46 × veste + 0.54 × piedino`. Il depth test non ha salvato
niente perché la geometria è arrivata nell'ordine sbagliato, il pezzo lontano
per primo. E non c'è modo di ordinarla: è un mesh rigido, non billboard
sortabili come quelli di `Flame.hpp`. Il back-face culling toglie il guscio
posteriore, non la geometria interna rivolta verso l'osservatore.

**La soluzione: due draw dello stesso mesh.** Non c'è nessuna divisione della
geometria tra i due shader — entrambe le passate disegnano il fantasma
**intero**, stessi indici (`Scene.hpp:557` e main.cpp:4980). Ogni frammento
passa due volte.

Prima passata, `PspectralDepth` + `SpectralDepth.frag`, compare **`LESS`**:

- piedino, `0.55 < 0.90` → passa, e scrive 0.55
- veste, `0.50 < 0.55` → passa, e scrive **0.50**

Il colore non cambia di un bit: la shader emette `vec4(0.0)`, e con
`srcAlpha × src + (1 - srcAlpha) × dst` (Starter.hpp:4439-4442) un alpha di 0
restituisce `dst`. È una color-write mask scritta con un blend factor, perché
`Pipeline` espone `setTransparency()` e non la mask. La depth invece viene
scritta lo stesso, `depthWriteEnable` è hardcoded a `VK_TRUE`
(Starter.hpp:4496). A schermo non è successo nulla; nel depth buffer c'è il
**minimo**, cioè la superficie del fantasma più vicina all'occhio. L'ordine di
arrivo non conta più: `LESS` scarta il lontano se arriva secondo e lo
sovrascrive se arriva primo.

Seconda passata, `Pspectral` + `Spectral.frag`, compare **`LESS_OR_EQUAL`**:

- piedino, `0.55 ≤ 0.50`? No → **scartato**, il fragment shader non gira nemmeno
- veste, `0.50 ≤ 0.50`? Sì → disegna

Un solo strato per pixel, il più vicino. Il piedino non viene coperto, viene
rifiutato dal depth test prima di poter contribuire al colore. Il criterio non è
mai "che pezzo sei", è sempre e solo "a che distanza sei rispetto al numero che
trovi nel pixel" — ed è **la veste stessa**, nella prima passata, ad aver
scritto il numero che poi ammazza il piedino nella seconda.

Il `LESS_OR_EQUAL` (main.cpp:2727) da difensivo diventa portante: con un `LESS`
stretto fallirebbe anche la veste, contro la depth che si è scritta da sola un
draw prima, e il fantasma sparirebbe del tutto. Il `LESS` del prepass invece non
è scritto da nessuna parte, è il default di `Pipeline::init`
(Starter.hpp:4298) — `PspectralDepth` non chiama mai `setCompareOp`.

**Dove stanno le due draw.** Il prepass deve cadere **dopo** il dungeon e
**prima** del colore dei fantasmi. Dopo il dungeon perché una depth di fantasma
scritta prima di un muro che gli sta dietro rifiuterebbe quel muro, lasciando un
buco a forma di fantasma; prima del colore per ovvi motivi. Ma
`Scene::populateCommandBuffer` percorre le technique una di fila all'altra senza
nessun aggancio in mezzo. Quindi lo slot che `Scene` gestisce è stato dato al
prepass — main.cpp:2937 registra `&PspectralDepth` come pipeline della technique
"Spectral" — e le draw a colori sono emesse a mano subito dopo
`SC.populateCommandBuffer()`, a main.cpp:4972-4981, iterando le stesse istanze
di `SC.TI[1]`. I descriptor set costruiti da `Scene` valgono per entrambe le
pipeline senza modifiche, perché sono costruiti sui DSL, che sono identici.

Costo: una seconda draw depth-only per fantasma, tre istanze, nessun descriptor
in più. `SpectralDepth.frag` non dichiara nemmeno un binding: solo il vertex
stage legge qualcosa, e il layout con cui la pipeline è creata è quello di
`Pspectral`.

> **Se il prof chiede**
>
> *"Perché serve un prepass se hai già il depth test?"* — Perché il depth test
> confronta il frammento in arrivo con **quello che c'è adesso** nel pixel, non
> con l'insieme dei frammenti futuri. Con la geometria che arriva dal lontano al
> vicino, il pezzo nascosto viene disegnato e poi il pezzo davanti gli si fonde
> sopra al 46%, lasciandolo visibile. Il prepass stabilisce il minimo prima che
> chiunque disegni colore, così il confronto avviene contro il valore giusto.
>
> *"Non bastava ordinare i triangoli?"* — È un mesh rigido: l'ordine corretto
> dipende dal punto di vista e cambia ogni frame. Ordinare per-triangolo in CPU
> costerebbe più delle due draw, e per geometria compenetrata non esiste comunque
> un ordine corretto.
>
> *"Perché due pipeline e non una riconfigurata?"* — Il compare op e il blending
> sono cotti dentro il `VkPipeline` alla `create()` (main.cpp:4728-4729):
> cambiarli richiede ricrearla, non si commutano per draw call.

## 3.15 Shader minori

- `UiQuad.vert` / `UiQuad.frag` — rettangoli a colore piatto. Colore via push constant, nessun descriptor set, nessuna texture. Esistono perché `TextMaker` sa disegnare solo glifi del suo atlante e nel range ASCII stampabile non c'è un rettangolo pieno con cui simulare uno sfondo. Servono al pannello della HUD cheat e al mirino (due istanze separate, con command buffer separati, perché disegnano contenuti indipendenti).
- `LightDebug.vert` / `LightDebug.frag` — croci e frecce colorate su ogni luce attiva, attivabili dal menu cheat.
- `framework/Text.vert` / `.frag` — il `TextMaker` del framework.
- `framework/ColliderShow.vert` / `.frag` — visualizzazione dei collider.

## 3.16 I file GLSL inclusi

GLSL di base non ha `#include`. Qui funziona perché CMake passa a `glslc` la
stessa `-I` del compilatore C++ e gli shader abilitano
`GL_GOOGLE_include_directive`.

- `custom/LightConstants.glsl` — incluso **sia da GLSL sia da C++** (`SceneLights.hpp`). Contiene solo direttive del preprocessore, che è l'unica sintassi su cui i due linguaggi concordano — e `MAX_LIGHTS` deve comunque essere una `#define`, perché dimensiona un array. Dentro: `MAX_LIGHTS`, `NUM_SHADOW_MAPS_2D`, `NUM_SHADOW_CUBES`, `SHADOW_CUBE_RES`, i tre tipi di luce, i bit `LIGHT_DEBUG_*`. Il punto è avere **una sola definizione** invece di due tenute in sync a mano.
- `custom/FlameColor.glsl` — `rgb2hsv` / `hsv2rgb` / `recolorStop`, condiviso da `Flame.frag` e `Spark.frag` così corpo e scintille si ricolorano allo stesso modo.
- `custom/Noise.glsl` — rumore condiviso.

> **Se il prof chiede**
>
> *"Spiegami i tre termini di Cook-Torrance."* — D: quante microfacce sono
> orientate lungo la half vector (GGX). G: quante di quelle sono nascoste o
> ombreggiate dalle vicine, importante agli angoli radenti. F: che frazione viene
> riflessa invece che trasmessa, Schlick su `V·H`, cresce fino a 1 all'orizzonte.
>
> *"Perché diffuso e speculare sono interpolati e non sommati?"* — Perché sommarli
> a piena forza restituirebbe più luce di quanta ne è entrata. `k` è la quota
> diffusa, `(1-k)` la speculare.
>
> *"Perché usi l'inversa trasposta per le normali?"* — Perché con una scala non
> uniforme la matrice di mondo non preserva la perpendicolarità. Nella nostra
> scena la strada è scalata [1,4,1].
>
> *"Perché il tone map sta nel composite e non nel fragment shader della scena?"*
> — Perché il bloom cerca i pixel sopra 1, e il tone map li schiaccerebbe dentro
> [0,1] prima che il bright pass possa trovarli.
>
> *"Come fai i filamenti che si staccano dalla fiamma?"* — Con il termine `carve`:
> il rumore può spingere il campo in negativo, e la sua ampiezza cresce con
> l'altezza, quindi vicino alla punta può ritagliare via pezzi della silhouette.
>
> *"Perché il domain warp e non semplicemente più ottave?"* — Più ottave variano
> la luminosità *sul posto*. Deformare il punto di campionamento fa arricciare e
> viaggiare le zone brillanti stesse, che è come si muovono le lingue di una
> fiamma vera.

---

# PARTE 4 — Le luci

`SceneLights.hpp` legge `assets/scenes/lights.json` all'avvio e trasforma ogni
voce in una `LightData`. Ogni frame `main.cpp` chiama `update()`, si fa
restituire la lista, la copia nel global UBO, e la GPU la manda agli shader. Lo
shader non legge mai un file e non vede mai questa classe: questo è l'unico
percorso da `lights.json` allo schermo.

I tre tipi (L09):

- **direct** — infinitamente lontana, quindi una sola direzione per tutta la scena e nessun decadimento con la distanza. Il sole.
- **point** — sta in una posizione e irraggia in tutte le direzioni, con decadimento. Una lampada, una candela.
- **spot** — una point ristretta a un cono. Ha una direzione di puntamento e due angoli.

Il decadimento è `color * (g / d)^beta`, dove `g` è la distanza alla quale la
luce vale esattamente `color` e `beta` l'esponente (0 costante, 1 lineare, 2
quadratico, che è quello fisicamente corretto).

In `lightRadiance()` la distanza è ammorbidita:

```glsl
const float NEAR_RADIUS = 0.4;
float distSoft = sqrt(dist*dist + NEAR_RADIUS*NEAR_RADIUS);
```

`max(dist, 0.0001)` da solo protegge la divisione ma `g/dist` cresce comunque
quasi senza limite avvicinandosi, quindi la radianza resta piuttosto piatta su
quasi tutta la stanza e poi schizza negli ultimi centimetri, dove il tone map la
schiaccia subito a bianco. Continuo sulla carta, ma si legge come "quasi non si
illumina, poi di colpo è al massimo" camminando verso un muro con la torcia in
mano. `NEAR_RADIUS` è un pavimento morbido su quanto `dist` può diventare
piccola (`sqrt(d² + r²)` non scende mai sotto `r`), circa la dimensione fisica
della fiamma: la stessa variazione totale di luminosità viene spalmata su più
distanza.

Per lo spot si moltiplica per il cono:

```glsl
cosAngle = dot(-L, lt.dir);
radiance *= clamp((cosAngle - cosOut) / (cosIn - cosOut), 0.0, 1.0);
```

`lt.dir` è dove il faro *punta*, quindi una lampada rivolta in basso è
`[0,-1,0]`; le slide scrivono la formula contro `lx`, da cui la negazione.

La posizione di una luce può essere in coordinate di mondo esplicite oppure
`instance` + `offset`, che la aggancia a un'istanza di `scene.json` così
sopravvive allo spostamento di quell'istanza. Stessa idea delle box di
collisione autorate.

**L'ambient emisferico.** `AmbientLight` ha `upper` (cielo), `lower` (terra),
`dir` (l'asse lungo cui i due si mescolano, cioè l'alto del mondo) e `weight`
(quota di luce indiretta, default 0.05, che è il valore *da interno*). Il
default è quello indoor perché il gioco si gioca dentro il dungeon: un modello
che non dichiara un peso è molto più probabilmente in una stanza chiusa che sotto
il cielo, e non va illuminato come se il soffitto sopra non facesse niente.

Siccome `CookTorrance.frag` fonde invece di sommare (§3.2), `weight` è la
*luminosità* del termine indiretto e `upper`/`lower` sono solo i suoi due
**colori** — motivo per cui in `lights.json` sono vicini a 1 e non vicini a 0.1
come erano quando venivano sommati.

Gli **switch di debug** sono per *tipo* di luce, non per singola luce, perché la
domanda a cui rispondono è "è il sole o una lanterna a fare quello?". Non
modificano `lights.json`: `update()` semplicemente non restituisce le luci il
cui tipo è disabilitato, quindi il file resta l'unica fonte di verità.

> **Se il prof chiede**
>
> *"Come modelli le tre luci?"* — Una struct sola con un `type`, e lo shader
> ramifica: la direct restituisce `color` senza decadimento, la point applica
> `(g/d)^beta`, la spot aggiunge il fattore del cono con interpolazione fra
> `cosIn` e `cosOut`.
>
> *"Cos'è l'ambient emisferico e perché non un ambient costante?"* — È due colori
> mescolati in base all'orientamento della superficie, quindi una superficie
> rivolta al cielo prende il colore del cielo e una rivolta al terreno quello del
> terreno. Un ambient costante appiattirebbe tutto.

---

# PARTE 5 — Le ombre

## 5.1 Come funziona lo shadow mapping, in generale

L'idea: renderizza la scena **dal punto di vista della luce**, salvando solo la
profondità. Il risultato dice, per ogni direzione dalla luce, quanto è lontana la
prima superficie colpita. Poi, mentre ombreggi normalmente, trasformi il
frammento nello spazio della luce e confronti: se il frammento è più lontano di
quello che la mappa ha registrato in quella direzione, c'è qualcosa in mezzo,
quindi è in ombra.

Il problema classico si chiama **shadow acne**: per errori di precisione e di
risoluzione una superficie finisce per fare ombra a se stessa, e si vedono
strisce scure. La cura tradizionale è un **bias**, cioè tollerare che l'occlusore
memorizzato sia un pochino più vicino prima di dichiarare l'ombra. Ma un bias
troppo grande produce **peter-panning**: l'ombra si stacca dall'oggetto, e
soprattutto la luce passa *attraverso* occlusori sottili. §5.6 racconta come
questo progetto risolve il conflitto.

**Come lo risolve davvero, sul percorso cubemap.** Ogni difesa tradizionale
contro l'acne è una forma di *slack* — un bias, un normal offset, una banda di
ammorbidimento — e lo slack è esattamente ciò che stacca l'ombra dal suo
occlusore: dove un occlusore sporge sopra la propria base il divario col
pavimento cresce lentamente, quindi anche pochi millimetri di tolleranza si
allargano in centimetri di pavimento illuminato. Tre giri di taratura di quel
numero, su due file diversi, hanno spostato la striscia senza mai chiuderla.

`PShadowCube.setCullMode(VK_CULL_MODE_FRONT_BIT)` **rimuove la premessa**: ogni
occlusore registra il lato girato DALL'ALTRA parte rispetto alla torcia, quindi
una superficie rivolta verso la luce non è nella mappa e non può fallire un
confronto con se stessa. La silhouette — l'insieme delle direzioni che
l'occlusore copre, l'unica cosa che decide dove cade l'ombra — è identica nei
due casi.

Il costo, ed è reale: un occlusore ora perde luce per il proprio **spessore**,
dato che ciò che viene registrato è la sua faccia lontana. È un errore limitato
dalla geometria e non da una costante, e ogni muro e ogni anta di porta di questo
dungeon è molto più spesso dello slack che sostituisce. Il caso da tenere
d'occhio è l'opposto: qualcosa costruito come una singola faccia piatta non ha
una faccia lontana da registrare, e smette di proiettare ombra dal lato che la
luce vede.

Il progetto ha **due famiglie** di ombre, una per tipo di proiezione.

## 5.2 Shadow map 2D — il sole

`NUM_SHADOW_MAPS_2D = 2` slot (oggi ne serve uno, l'altro è margine per una
futura spot che proietti ombra). Un render pass depth-only `AT_DEPTH_ONLY`
ciascuno, a 1024×1024, e **una sola pipeline condivisa**: i render pass hanno
configurazione di attachment identica, quindi vale la regola di compatibilità
(§0.4). Sono creati una volta in `localInit()` e mai toccati da un resize —
un target offscreen non dipende dalla finestra.

Il sole usa una proiezione **ortografica** (i raggi sono paralleli), e questo
rende il bias facile: la sua box distribuisce 1..200 linearmente, quindi
`0.0015` fisso vale ~30 cm ovunque.

In `shadowFromMap2D()` ci sono due controlli che vale la pena saper spiegare:

- `if(lightClip.w <= 0.0) return 1.0;` — per una matrice prospettica `w` è la distanza in *avanti* dalla camera, quindi `w <= 0` significa che il punto è dietro. La divisione prospettica specchierebbe un punto del genere dentro il range 0..1 della mappa, facendo campionare una profondità che appartiene a una direzione completamente diversa. Il sole, essendo ortografico, dà sempre `w = 1` e non ci passa mai.
- il controllo che le UV siano dentro 0..1 — fuori dalla box della mappa non c'è niente contro cui confrontare, e senza il controllo si campionerebbe spazzatura sul bordo clampato.

Nota su `lightNDC.z`: grazie a `GLM_FORCE_DEPTH_ZERO_TO_ONE` (impostato in
`Starter.hpp`) la z è già nel range 0..1 di Vulkan, lo stesso in cui è
memorizzata la mappa. Solo le XY vanno rimappate da -1..1 a 0..1.

## 5.3 Cube shadow map — le torce

Una point light irraggia in **tutte** le direzioni: una singola mappa
prospettica ne copre al massimo un emisfero. La soluzione standard è una
**cubemap**: sei mappe a 90° che insieme coprono la sfera.

Ogni slot è una vera immagine cubemap: un `VkImage` a **6 layer** creato con
`VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT`, formato `R32_SFLOAT`, 1024 per faccia. Ci
sono due tipi di view sopra:

- **una** view `VK_IMAGE_VIEW_TYPE_CUBE` su tutti e sei i layer, quella che `CookTorrance.frag` campiona con `samplerCube`;
- **sei** view a layer singolo, una per faccia, per poterci renderizzare dentro una faccia alla volta.

`NUM_SHADOW_CUBES = 32` slot.

Come già detto in §3.6, ogni faccia memorizza la **distanza lineare**, non la
profondità. Questa scelta ha un secondo vantaggio: una cubemap depth-only (sei
proiezioni prospettiche indipendenti) richiederebbe un bias diverso per faccia e
per angolo per restare priva di artefatti — esattamente il problema che aveva
costretto il vecchio workaround a due mappe per torcia a tarare due campi visivi
a mano. Una distanza lineare è la stessa unità in ogni direzione, quindi basta
un solo bias in unità di mondo.

## 5.4 `RPShadowCubeCompat` — la stranezza da saper spiegare

`RenderPass::createRenderPass()` in `Starter.hpp` è **privato**, e i 36
framebuffer per-faccia devono essere costruiti a mano, perché attaccano view a
singolo layer dentro un'immagine a 6 layer — cosa che `FrameBufferAttachment` non
sa fare (crea sempre una `VK_IMAGE_VIEW_TYPE_2D` a 1 layer).

Per creare un framebuffer serve un `VkRenderPass` compatibile. L'unico modo di
ottenerne uno restando dentro l'API pubblica della classe è far costruire a un
`RenderPass` completo il suo, e rileggerne il campo `.renderPass`.
`RPShadowCubeCompat` esiste solo per questo: la sua immagine e il suo framebuffer
(1 layer, 1024) non vengono mai usati. È burocrazia inevitabile per non
duplicare `vkCreateRenderPass` a mano.

## 5.5 Il pool dinamico e il caching — la parte più interessante

Ci sono più point light degne di un'ombra che slot disponibili (sei torce a
muro, la torcia decorativa della sala dl, due candele, la torcia in mano...).
Quindi gli slot **non** sono assegnati al momento del caricamento:

- l'**ultimo** slot (`HAND_TORCH_SHADOW_INDEX = NUM_SHADOW_CUBES - 1`) è riservato alla torcia in mano, che non passa da `lights.json`: la sua matrice di mondo non esiste in forma sensata finché `GameLogic()` non comincia a riscriverla ogni frame, quindi non può ricevere un indice fisso a load time né matrici fisse in `computeShadowMatrices()`. `updateHandTorchShadow()` gliele ricalcola ogni frame.
- **tutti gli altri** li distribuisce `updateDynamicShadowSlots()` a runtime, alle luci attualmente più vicine al giocatore, ogni `SHADOW_REASSIGN_INTERVAL = 0.3 s` (i candidati sono statici, si muove solo il giocatore: non serve una risposta per frame).

Vale la pena saperlo se il prof chiede quanto pesa: **in questa scena il contest
non scatta mai**. La passata sugli slot vuoti è incondizionata e i candidati sono
12 contro 31 slot dinamici, quindi ognuno tiene la sua ombra in permanenza. La
logica di contesa qui sotto serve a un livello autorato con più point light
degne d'ombra che slot liberi.

Con **isteresi**: un candidato in attesa deve essere più vicino dell'occupante
attuale di un fattore `SHADOW_SWAP_MARGIN = 1.15` per prendergli lo slot. Senza
margine, un giocatore fermo sul confine di distanza fra due candidati farebbe
cambiare slot — e quindi forzerebbe un re-render completo dell'ombra, dato che
non esiste crossfade fra "ha un'ombra" e "non ce l'ha" — praticamente a ogni
rivalutazione.

**Il caching.** `lastRenderedOccupant[t]` ricorda per quale "identità occupante"
lo slot `t` ha davvero renderizzato le sue sei facce. Ogni frame si confronta con
l'identità attuale (32 confronti fra interi, gratis); in caso di mismatch lo slot
finisce in `pendingCubeSlotRenders` e viene renderizzato **una volta sola**
tramite un command buffer usa-e-getta (`renderCubeSlotsOnce()`), fuori dal
command buffer "main".

Il motivo è §0.6 al contrario: **tutto ciò che viene registrato nel main buffer
si ridisegna a ogni frame per sempre**, per quanto raramente il suo contenuto
cambi. Con il caching, le point light statiche — che in questa scena sono tutte
tranne la torcia in mano — vengono renderizzate **esattamente una volta per la
vita del programma**, invece delle 9000+ draw call per frame complessive che
costava il vecchio loop incondizionato.

`SHADOW_SLOT_UNSET = -2` è lo stato iniziale di ogni slot. Ogni slot ha bisogno
di **almeno un render** anche se resta vuoto per sempre: altrimenti la sua
immagine resta in `VK_IMAGE_LAYOUT_UNDEFINED`, contro cui un descrittore
`samplerCube` non può legalmente essere bindato. Da qui il percorso
"begin/end e basta" in `recordCubeSlotFaces()` per uno slot non occupato.

**La sottomissione separata (`submitCubeShadowCaptures`).** Le catture non
stanno nel command buffer "main": quello è registrato una volta per immagine
della swapchain e riprodotto identico, e non sa esprimere "questi slot, questo
frame". Vengono quindi registrate e sottomesse a parte, alla fine di
`updateUniformBuffer()` — dopo che ogni istanza ha la matrice di mondo del frame
corrente, perché questa passata binda **lo stesso descriptor set per-istanza**
della passata principale, e registrarla prima cuocerebbe una `Wm` stantia dentro
la mappa.

Due cose rendono quella sottomissione separata *corretta* e non solo comoda, ed
è la parte da saper difendere:

- Viene sottomessa **prima** del buffer principale, sulla stessa coda, quindi la precede nell'ordine di sottomissione. Una **barriera** in fondo ordina poi queste scritture di colore rispetto a ogni comando successivo sulla coda, campionamento delle cubemap incluso. È questo che fa vedere alle letture delle catture finite, ed è ciò che ha sostituito il `vkQueueWaitIdle` che c'era prima: la GPU non deve più svuotarsi a metà frame, deve solo ordinare due stadi fra loro.
- Ogni immagine della swapchain ha il **suo** buffer e la **sua** fence, attesa prima di ri-registrare, perché ri-registrare un buffer ancora in volo è undefined behaviour. A regime quella fence è segnalata da un pezzo e l'attesa ritorna subito.

**Il cull per faccia (`cullPerFace`).** Nel renderizzare una faccia si scartano
le istanze che le cadono fuori. Una faccia copre un sesto della sfera, quindi il
taglio è circa 6×, ed è ciò che rende sostenibile una ri-cattura per frame.

Ha però un vincolo di correttezza che vale la pena saper enunciare: **è valido
solo per un command buffer registrato e sottomesso nello stesso frame**. Un
insieme di visibili deciso al momento della registrazione smette di essere vero
appena la luce o gli occlusori si muovono, quindi un buffer riprodotto su più
frame — il buffer "main" di §0.6 — non può usarlo. Il flag esiste invece di
essere dato per scontato proprio per rendere esplicito quel vincolo dove viene
sfruttato.

**L'invalidazione per faccia (`faceMask` / `pendingFaceMask`).** Il caching
sopra ragiona per slot: o si ridisegnano tutte e sei le facce o nessuna. Ma un
occlusore che si muove — una porta che si apre, un fantasma che passa — invalida
tipicamente **una faccia sola**. `movingOccluders` marca in `pendingFaceMask`
solo le facce toccate, e `recordCubeSlotFaces()` salta le altre: una faccia fuori
dalla maschera non viene né pulita né iniziata, e conserva i texel dell'ultimo
disegno.

Il guadagno è concreto: una cubemap è sei immagini indipendenti che condividono
un handle, e a 1024² R32 per faccia **i clear costano più dei draw**. Ridisegnare
tutte e sei per un fantasma che ne tocca una o due significava pulire e riempire
6 MB di render target per sistemarne 1. Uno slot la cui LUCE è cambiata prende
comunque tutte e sei (`ALL_CUBE_FACES`): niente del vecchio contenuto sopravvive
allo spostamento della camera da cui è stato ripreso.

Una faccia viene marcata in due casi: contiene un mover che si è spostato da
quando la faccia è stata disegnata l'ultima volta, oppure ne conteneva uno alla
cattura precedente e ora non più (i suoi texel hanno ancora addosso un fantasma
che se n'è andato). Tutto il resto costa un test di distanza e uno di frustum.
La portata usata è quella **propria di ogni luce**, misurata contro la sfera
avvolgente del mover e non contro la sua origine, così le uniche catture saltate
sono quelle il cui risultato nessuno potrebbe vedere. Non c'è budget per frame
né coda: un'ombra aggiornata con un frame di ritardo è un'ombra che si vede
inseguire il suo fantasma.

Corollario da ricordare: una faccia deve entrare nella maschera **almeno una
volta** prima che qualcuno campioni il cubo, per lo stesso motivo di
`VK_IMAGE_LAYOUT_UNDEFINED` di sopra. Il diff sugli occupanti se ne occupa
chiedendo `ALL_CUBE_FACES` la prima volta che vede ogni slot.

## 5.6 Il filtraggio del cubo — `shadowFromCube()`

Il confronto grezzo `(dist - bias > closestDist) ? 0.0 : 1.0` è un test binario
acceso/spento, che su un muro si legge come un bordo tagliente come un rasoio.

**PCF non è stato usato.** Il PCF (Percentage Closer Filtering) consiste nel
campionare in più direzioni jitterate e mediare. È stato provato e rimosso:
`sampleShadowCube()` non è una `texture()` semplice ma una **catena di branch su
32 binding**, quindi ogni tap ripete tutta la catena. Con i pochi tap che ci si
poteva permettere, i campioni cadevano abbastanza lontani da leggersi come
chiazze separate sovrapposte invece che come un bordo morbido.

Al suo posto, la stessa morbidezza con **un solo campione**:

- **rampa invece di soglia**: si passa da pienamente illuminato a pienamente in ombra su una piccola banda di distanza di mondo, che comincia esattamente dove stava la soglia del test netto. Importante: la rampa parte da `occluderGap == bias`, non da 0, così esiste un **plateau piatto `lit == 1.0`** per ogni `occluderGap <= bias`. Con la formulazione precedente una superficie non occlusa finiva solo `bias` unità dentro la rampa invece che saldamente sul plateau, quindi i muri normalmente illuminati stavano a metà rampa e qualunque rumore di precisione fra un texel e l'altro poteva far cadere il risultato da una parte o dall'altra — ed è esattamente il "di colpo non è illuminato" che si vedeva.
- **normal offset invece di un bias grande**. Questa è la parte da raccontare bene.

Il ragionamento sul normal offset: un texel di una faccia della cubemap copre
`2*dist/SHADOW_CUBE_RES` di spazio di mondo alla distanza `dist` (una faccia
copre 90°, quindi la sua larghezza a distanza `d` è `2d`). Su quel texel la
distanza registrata della superficie varia di quella larghezza per la **tangente
dell'angolo di incidenza** — ed è questo che fa sì che una superficie radente
abbia bisogno di più tolleranza di una frontale.

Quell'errore però **non deve essere pagato con il bias di profondità**. Un bias è
una distanza a cui un occlusore reale può stare davanti a una superficie senza
fermare la luce. Nel nostro dungeon il pannello di una porta sta ~0.65 unità
davanti alle catene inchiodate sopra: appena il bias si avvicina a quel valore,
**la porta smette di essere una porta** e la torcia dall'altro lato illumina le
catene attraverso il legno. E il bias cresce con la distanza (il texel cresce)
mentre quel divario no, quindi nessun cap funziona: alle catene a 5 unità
teneva, alle torce a 10 e 12 arrivava al suo tetto e passava — ed erano
esattamente i due colori che si vedevano trapassare.

Il **normal offset** spende la stessa quantità in una direzione dove non costa
niente: sposta il punto di lookup lungo la normale della superficie, fuori dalla
superficie e verso il lato della luce, così finisce in un texel la cui distanza
registrata appartiene davvero a *questa* superficie e non al tratto di essa mezzo
texel più in là. La soglia di profondità resta piccola e indipendente dalla
distanza, quindi lo spessore di una porta la batte sempre.

Numeri: bias `0.02..0.06`, offset cappato a `0.12` (anche l'offset va cappato:
spinto abbastanza lontano lungo la normale il punto di campionamento finirebbe
oltre un occlusore che sta proprio davanti, che è la stessa perdita per un'altra
strada).

## 5.7 Il far plane, e una lezione

`TORCH_SHADOW_FAR_CONST = 60.0`. Vale la pena conoscerla perché è un bug reale
diagnosticato male all'inizio.

Il clear value dell'attachment è il far plane stesso, così "niente disegnato"
si legge come "occlusore infinitamente lontano" = illuminato. Questo funziona
**a patto che** `shadowFromCube()` non venga mai interrogata oltre quel piano.

Era vero quando il far era 15: con la vecchia taratura di `g`/`beta` la torcia
era già scesa a pochi punti percentuali a 15 unità, e finiva sotto
`LIGHT_ATTEN_EPS` poco dopo. L'invariante si è rotta quando il falloff è stato
ritarato per una portata più lunga: la torcia restava visibilmente brillante ben
oltre le 15 unità, quindi ogni muro più lontano di quello **veniva** interrogato
— e si riprendeva il clear value come "occlusore più vicino", che è più vicino
della sua distanza reale, quindi si leggeva come falsamente in ombra.

Il sintomo sembrava "la luce della torcia arriva solo fino a un raggio fisso, con
un bordo netto lì". Era invece un bug delle ombre. Alzato a 60 per coprire
comodamente le ~60 unità di footprint del dungeon.

## 5.8 Chi proietta ombra

Solo le istanze della tecnica CookTorrance vengono disegnate negli shadow pass, e
non tramite `Scene::populateCommandBuffer` (che percorrerebbe ogni tecnica,
fiamme incluse) ma con un loop scritto a mano che itera `SC.TI[0].I[j]`.

Le fiamme sono deliberatamente escluse: non sono occlusori, ed essendo
traslucide non dovrebbero occludere. Escluderle è più semplice che dare loro una
logica di ombra propria.

Anche i corpi illuminanti sono esclusi, tramite `Material::castsShadow` letto da
`materials.json`: è lo stesso lookup `forModel()` che il main pass fa comunque
per il BRDF, quindi non costa niente oltre il branch.

## 5.9 Il sampler che continua a mordere

Vale la pena raggruppare i tre punti in cui lo stesso dettaglio ha causato un
bug, perché è una bella risposta se ti chiedono di raccontare un problema
concreto: `Starter.hpp` costruisce il sampler di ogni framebuffer attachment con
l'address mode di default, che è **REPEAT**. Conseguenze:

- in `BloomBright.frag`, i quattro tap che escono da un bordo leggono il bordo opposto;
- in `BloomBlur.frag`, idem per i nove tap;
- in `Composite.frag`, perfino senza offset l'impronta bilineare dell'upsample esce nell'ultimo mezzo texel.

Il sintomo era sempre lo stesso: una banda o una linea brillante in cima o in
fondo allo schermo ogni volta che qualcosa di sovraesposto stava all'altro capo.
Risolto clampando le coordinate a mano dentro le tre shader — equivalente a
`CLAMP_TO_EDGE` — invece di cambiare un default di sampler su cui ogni texture
del livello conta per il tiling delle UV.

Per le cubemap invece si usa un sampler nostro (`cubeShadowSampler`,
CLAMP_TO_EDGE) e filtraggio **NEAREST**: il bilineare sulla cubemap
interpolerebbe fra distanze appartenenti a superfici diverse, producendo valori
intermedi che non corrispondono a niente di reale, ed è ciò che il vecchio bias
sovradimensionato stava in realtà compensando.

Vale la pena avere pronto il dettaglio, perché è la parte che si spiega meglio.
Attraverso una silhouette — un texel sull'anta della porta e quello accanto che
guarda oltre il bordo fino al muro in fondo — il bilineare restituisce una
distanza che non appartiene a **nessuna** delle due. L'errore è proporzionale al
salto di profondità attraverso il bordo, cioè **metri, non texel**, ed è per
questo che serviva un bias dello stesso ordine (0.35) per mascherarlo. Quel bias
è esattamente ciò che permetteva a una torcia di illuminare le catene
**attraverso una porta chiusa**, dato che la porta sta solo ~0.6 unità davanti a
loro. Campionare un texel solo rende il confronto onesto e riporta il bias a
essere una quantità dell'ordine del texel.

> **Se il prof chiede**
>
> *"Perché una cubemap per le point light e non una shadow map normale?"* — Una
> mappa prospettica copre al massimo un emisfero, una point irraggia in tutte le
> direzioni. La cubemap copre la sfera con sei facce a 90°.
>
> *"Perché memorizzi la distanza e non la profondità?"* — Perché la lookup su
> `samplerCube` è per direzione e non sa quale faccia ha risposto, quindi il
> valore deve significare la stessa cosa su ogni faccia. Vale per una distanza
> euclidea, non per una profondità prospettica.
>
> *"Come gestisci lo shadow acne?"* — Sul percorso cubemap, **togliendo la
> premessa invece di compensarla**: `PShadowCube` culla le facce ANTERIORI, quindi
> una superficie rivolta verso la luce non è nella mappa e non può confrontarsi
> con se stessa. Non c'è acne da pagare. Quel che resta è un bias minuscolo
> (0.0015..0.004) che copre solo il rumore in virgola mobile fra la distanza
> calcolata qui e quella calcolata in `ShadowCube.frag`, più il PCF. Il normal
> offset **non c'è più**: `NORMAL_OFFSET_TEXELS` è 0. Serviva a centrare la
> lookup nel texel della superficie giusta, problema che esiste solo se la
> superficie è nella mappa, e costava uno spostamento laterale del bordo
> d'ombra. Sul sole, ortografico, resta il bias fisso `0.0015` (§5.2).
>
> *"Hai 32 cubemap, non è tantissimo?"* — Non sono tutte attive: sono un pool
> assegnato a runtime alle torce più vicine al giocatore, con isteresi per non
> farlo oscillare. E vengono renderizzate una sola volta ciascuna finché il loro
> occupante non cambia.

---

# PARTE 6 — Le torce e le fiamme

L'elemento più stratificato del progetto. Una "torcia" è la composizione di
cinque cose distinte.

## 6.1 L'istanza di scena

Un normalissimo modello di `scene.json` (`dungeonTorch`, `dungeonTorchHeld`,
`dungeonCandle`). La torcia in mano è un'istanza come tutte le altre, la cui
matrice di mondo viene riscritta ogni frame da `GameLogic()` a partire dalla posa
della camera: `HAND_TORCH_OFFSET` in spazio camera (destra, su, avanti), più
un'inclinazione extra perché sembri impugnata, più una scala uniforme (il modello
`SM_Torch_01` è dimensionato per un supporto a muro).

Curiosità utile: `SM_Torch_01` (a muro) e `SM_Torch_Held_01` (in mano) si sono
rivelate essere la stessa identica mesh.

## 6.2 La registrazione data-driven — `flames.json`

Non è hardcodata per istanza: è chiavata **per modello**. Ogni istanza che usa
una delle mesh elencate riceve automaticamente una fiamma e una luce, senza
modifiche a `main.cpp` e senza autorare niente per istanza. Un livello nuovo che
riusa le mesh standard ottiene le sue fiamme gratis.

Per ogni modello:

- `anchor` — il punto, **in spazio locale del modello**, dove sta lo stoppino. Ricavato camminando l'accessor POSITION della mesh (per la candela: il tappo di cera si chiude a Y 0.618..0.638, poi c'è spazio vuoto perché il fianco del lucignolo non è modellato, poi la punta si chiude a Y 0.731..0.734; l'ancora sta a 0.70, in mezzo a quel vuoto).
- `sizeScale`, `lightScale` — default 1.0.
- `isCandle` — cambia solo quale toggle del menu debug la governa.
- `overrides` per istanza, per i pochi casi che la mesh da sola non può determinare: un'istanza che vuole colore, dimensione o stato acceso/spento diversi da quelli del suo modello. Oggi è vuoto: conteneva le quattro torce colorate della sala dl (rossa, verde, blu, viola), rimosse perché quattro sorgenti sature sulla stessa parete davano a ogni oggetto della stanza quattro ombre sovrapposte, ciascuna tinta dalle luci che ancora la raggiungono. Al loro posto c'è una singola torcia arancione normale (`dlTorch` in `scene.json`).

La torcia in mano è l'unica esclusa, pur usando `dungeonTorchHeld`: `main.cpp` la
spawna per id letterale (`handTorch`) prima ancora di leggere il file, perché è
un singleton di gameplay (ce n'è sempre e solo una) e non arredamento di livello,
e ha bisogno del cablaggio `heldByCamera` che nessun'altra fiamma ha.

## 6.3 L'inviluppo del fuoco, simulato in CPU

`fireFbm()` (main.cpp:244): tre ottave di value noise 1D costruito su un **hash
intero**. Non il classico `fract(sin(x)*43758.5)`, che è notoriamente
dipendente dal driver in GLSL e che in C++ a doppia precisione non decorrela
abbastanza bene agli input piccoli usati qui. Tre ottave sono abbastanza perché
nessuna singola frequenza si senta nel risultato, e poche abbastanza da restare
un errore di arrotondamento accanto al resto del frame.

**Perché in CPU e non nello shader:** il flicker deve essere condiviso da tre
consumatori — la fiamma, le sue scintille e la point light che getta — e solo la
CPU li vede tutti e tre. Una luce non può comunque sfarfallare per-pixel.

Due inviluppi **separati** dallo stesso segnale, ed è la separazione che ha
eliminato i "salti" dell'intera fiamma:

- `intensity` — la **luminosità**, ~0.30..1.40, inseguita con una molla, veloce;
- `heightScale` — l'**altezza**, ~0.78..1.09, inseguita con un polo singolo molto più lento (`FLAME_HEIGHT_TAU`).

La giustificazione fisica: l'emissione luminosa può guizzare in fretta, ma
l'altezza segue la colonna di combustibile e resta indietro. Una fiamma che cala
è visibilmente **più bassa**, non solo più fioca, esattamente come una vera a cui
manca il combustibile o presa da uno spiffero — per questo `heightScale` scala
tutta l'altezza del quad e non solo colore e alpha.

Sopra ci sono tre modulazioni:

- **guttering** — collassi brevi e irregolari in cui la fiamma si accuccia e si smorza.
- **stare-at glare** — avvicinarsi a una torcia a muro e centrarla in vista alza `glareBoost` per quella fiamma e l'esposizione globale. Lo smorzamento è **asimmetrico**: l'abbaglio arriva in fretta, l'occhio recupera più lentamente. L'asimmetria è anche ciò che impedisce il pumping quando lo sguardo va e viene lateralmente.
- **lean** — la velocità della mano, smorzata (una `TAU` che decide quanto in fretta la velocità smorzata insegue quella vera), convertita in mezze larghezze di offset della punta per unità-di-mondo/secondo, con un tetto duro: una fiamma che si trascina più indietro di così smette di leggersi come fuoco trascinato e comincia a sembrare un oggetto rigido.

Il flicker veloce e per-pixel **non** è qui: sta in `Flame.frag` come shimmer
(§3.7 punto 10).

## 6.4 La luce proiettata

Una point light con colore e falloff comuni a tutte le torce, moltiplicata
dall'inviluppo `intensity` così fiamma e luce si smorzano insieme. Le candele
riducono ulteriormente `g`, cioè la loro portata.

`TORCH_LIGHT_CULL_DIST` limita quante torce vengono caricate nel global UBO e a
che distanza. La priorità di un candidato è la **distanza al quadrato dall'occhio,
gonfiata per tutto ciò che non è davanti alla camera**: una luce dietro le spalle
conta meno di una in vista, ma non viene azzerata, così una molto vicina alle
spalle resta comunque credibile. Lo stesso criterio pesa la scelta degli slot
ombra.

**Perché il cull delle luci conta più di quello della geometria.**
`CookTorrance.frag` cicla su ogni luce per ogni frammento, e con il per-sample
shading forzato (§0.9) questo significa una valutazione GGX completa **quattro
volte per pixel** a 4× MSAA, che la luce si veda o no. È il costo GPU dominante
del renderer, ben sopra il cull geometrico, che taglia solo draw call — e le draw
call non sono mai state il collo di bottiglia a questo numero di istanze.

Il raggio è passato da 25 a 120 e ora a 55, e la storia è istruttiva. Il dungeon
è largo ~60 unità, quindi 25 spegneva le torce di qualunque stanza in cui il
giocatore non fosse — **visibilmente**, perché il billboard della fiamma è
incondizionato: restava acceso a schermo mentre non illuminava nulla, con le
pareti attorno nere. È esattamente il caso che una stima puramente numerica
("contribuisce il 3%, sotto la soglia che l'ambient nasconde") non riesce a
prendere.

**Il cull geometrico (`GEOM_CULL_*`)**: un raggio attorno al giocatore, più un
cono più lungo lungo la direzione di sguardo. Sostituisce un approccio a grafo di
stanze (flood fill attraverso le porte aperte) che con questo asset pack era
fragile: i pezzi di muro non piastrellano in modo affidabile l'impronta completa
di una stanza, quindi le box derivate avevano buchi grandi abbastanza da
nascondere la stanza in cui il giocatore *stava*. Il test raggio+cono legge solo
la posizione e il facing della camera e la posizione di ogni istanza: nessuna
topologia da sbagliare, niente da riautorare quando il livello cambia.

Il compromesso è voluto: non sa che c'è un muro in mezzo, quindi un'istanza
appena oltre una porta aperta può comparire poco prima che la porta sia
raggiunta. È un taglio approssimato di "cosa vale la pena disegnare", non un
sistema di visibilità portal-correct.

**La nebbia è agganciata allo stesso numero.** `gubo.fogDensity` non è una
costante tarata a mano: risolve `exp(-(density·dist)²) = 0.01` per
`dist = GEOM_CULL_CONE_DIST × 2.5`, così nebbia e cull non possono divergere. Il
fattore 2.5 è l'unica manopola. A 1.0 la nebbia arrivava a saturazione
*esattamente* alla distanza di cull, e si leggeva come troppo scura molto prima
— una curva esponenziale perde luminosità in fretta assai prima di "arrivare" al
suo residuo. A 2.5 la vicinanza resta entro pochi punti percentuali dello
scoperto fino all'anello `GEOM_CULL_RADIUS`, e si è a circa metà luminosità alla
portata massima del cono. Il prezzo è che un po' di pop geometrico può affacciarsi
proprio sul bordo, mitigato dalla vignette e dal fatto che nebbia e sfondo
condividono lo stesso nero.

I due cull sono **legati da uno `static_assert`** che impone
`TORCH_LIGHT_CULL_DIST >= GEOM_CULL_CONE_DIST` (oggi 55 e 50). Così qualunque
cosa il cui supporto-torcia è ancora disegnato è garantita ancora illuminata, e
il caso "torcia visibile ma al buio" non può ripresentarsi. Abbassare il cull
delle luci sotto quello geometrico non compila.

## 6.5 La fiamma

**Come è costruita la base del billboard.** Le fiamme sono quad orientati verso
la camera, e la base viene costruita in CPU una volta per frame e moltiplicata
per la ViewPrj (§2.7). Ce ne sono **due**, e il perché è una bella domanda
d'esame.

La prima è **cilindrica**, condivisa da tutte le fiamme del mondo, e prende
l'asse *right* **della camera** invece del vettore occhio→ancora di ogni fiamma.
Il billboard cilindrico da manuale userebbe `cross(worldUp, eyePos - anchor)`, e
funziona per una torcia a muro a qualche unità di distanza. Cade sulla torcia in
mano: quell'ancora sta a meno di un'unità dall'occhio ed è piazzata in spazio
camera, quindi inclinando lo sguardo ruota in blocco attorno all'occhio. Il suo
offset **orizzontale** dall'occhio tende a zero e, alla pitch in cui la torcia
passa esattamente sopra (o sotto) la camera, cambia segno: lo yaw derivato fa un
salto di 180°, cioè la fiamma **gira su se stessa** quando guardi in su o in giù,
e appena prima del salto quel vettore è quasi di lunghezza nulla, quindi la sua
direzione è già rumore numerico.

L'asse right della camera non ha nessuno dei due problemi: la camera è costruita
yaw-poi-pitch senza roll, quindi quell'asse è esattamente orizzontale a ogni
pitch, non degenera mai, e ruota solo con lo yaw — che è l'unica rotazione che
una fiamma in piedi deve seguire. Orientarsi verso il **piano** di vista invece
che verso il **punto** di vista è comunque la scelta standard: è ciò che impedisce
a un billboard di ruotare lentamente mentre scorre attraverso lo schermo, e alle
distanze di una torcia le due cose sono indistinguibili.

La seconda base è **solo per la torcia in mano**, e usa gli assi veri della
camera, pitch compreso, letti dalle colonne di `camToWorld`. Quella torcia non è
un oggetto del mondo davanti a cui la camera passa: è saldata alla camera e si
inclina visibilmente con lo sguardo. Una fiamma che resta verticale nel mondo
sopra un manico che si inclina esce dall'allineamento, e oltre una pitch modesta
sembra sporgere di lato invece che bruciare sulla punta — peggio della rotazione
che la base cilindrica serviva a togliere. Legandola alla stessa base che cavalca
il mesh della torcia, le due restano rigide a ogni pitch, e poiché la base è
letta direttamente dalla matrice di vista non può degenerare né ribaltarsi.

Vedi §3.7 e §3.8 per gli shader. Il punto architetturale da ripetere perché è
quello su cui vale la pena essere chiari: **il bagliore attorno alla fiamma non è
geometria**. Non c'è un billboard di glow, non c'è un alone modellato, non c'è un
secondo pass. La fiamma scrive valori HDR reali sopra 1 nel target floating
point, e la catena di bloom li trasforma in alone. Stessa cosa per ExitGlow e per
il focus glow.

Le fiamme sono disegnate **dentro il main pass**, subito dopo la geometria
(`flame.populateCommandBuffer()` chiamata subito dopo
`SC.populateCommandBuffer()`), non in un render pass separato come serve invece a
`UiQuad`. È così che fanno depth test contro il castello gratis, nel modo
normale.

E sono **escluse da tutti gli shadow pass** (§5.8).

Nota sul dimensionamento: `maxInstances` di `Flame::init` è fisso al momento
dell'init, come `MAX_LIGHTS`, perché i descriptor set vengono da un unico pool
condiviso la cui dimensione va dichiarata prima di crearlo. "Quante fiamme
esisteranno mai" deve essere noto in anticipo, non può crescere a richiesta.

## 6.6 Le candele

Sono il terzo interagibile, accanto a porte e pickup. Hanno un flag `burning` a
runtime: una candela spenta non ha fiamma e non ha luce.

Si accende guardandola e premendo E, ma **solo se il giocatore ha del fuoco in
mano** — che oggi significa la torcia, che è un'istanza di scena sempre presente.
Il raggio è misurato in 3D come per un pickup (non solo in XZ come per una porta,
perché una candela sta su un tavolo e l'altezza conta). Il cono di mira è stretto
quanto quello di un pickup: il lucignolo è un bersaglio piccolo.

L'aura di focus è arancione fuoco (`glow = 3`), scelta più calda e più rossa
dell'oro delle porte così le due si distinguono a colpo d'occhio: una candela
promette **fuoco**, e l'aura è l'unico indizio che il giocatore riceve prima di
premere il tasto. Diventa rossa se non c'è niente con cui accenderla.

> **Se il prof chiede**
>
> *"L'alone attorno alla fiamma è geometria?"* — No. La fiamma scrive valori sopra
> 1 in un target HDR e il bloom fa il resto. Non c'è nessun billboard di glow.
>
> *"Perché il flicker è calcolato sulla CPU e non nello shader?"* — Perché lo
> devono condividere fiamma, scintille e point light, e solo la CPU li vede tutti
> e tre. Il flicker veloce per-pixel invece è nello shader, perché una luce non
> può sfarfallare per-pixel.
>
> *"Le scintille sono un particle system?"* — No, sono completamente procedurali:
> il numero è fisso nella mesh e ogni scintilla ha un seed da cui derivano
> spawn, traiettoria e durata, in loop con `fract()` sul tempo. Il "rate" è
> simulato con soglie per-scintilla contro l'inviluppo della fiamma.

---

# PARTE 7 — Il gameplay

## 7.1 Il ciclo di caccia

`HuntCycle.hpp` possiede **solo** l'orologio e un valore di blend del colore. Non
sa cosa sia una torcia, non tocca le luci, non guarda mai un fantasma.
`main.cpp` gli chiede "di che colore dev'essere una fiamma" e "i fantasmi devono
cacciare", e fa il lavoro da sé. Stessa divisione di `SceneLights`, che costruisce
i `LightData` e lascia il caricamento a `main.cpp`.

Tre fasi in loop, tutte tarabili da `gameplay.json`:

- **Calm** (45 s) — le torce bruciano dell'arancione autorato, i fantasmi camminano i loro giri di pattuglia e ignorano completamente il giocatore.
- **Warning** (3 s) — un telegrafo breve. Le fiamme virano verso il colore della caccia e pulsano, ma i fantasmi sono ancora in pattuglia. Questa fase esiste **solo** perché essere cacciati non sia mai una sorpresa: il giocatore ha un paio di secondi per trovare una porta o un angolo prima che qualcosa si muova verso di lui.
- **Hunt** (18 s) — colore pieno, ogni fantasma molla la pattuglia e viene addosso. Toccarne uno chiude la run.

Dopo, `recoverDuration` (4 s) di ritorno all'arancione **mentre la fase è già
Calm**: il pericolo finisce nell'istante in cui la fase cambia, la dissolvenza
serve solo a non far scattare la stanza all'arancione in un frame. È per questo
che il blend del colore è un valore a sé e non qualcosa derivato dalla fase.

Calm lungo e Hunt corto è deliberato: una caccia è un'interruzione
all'esplorazione, e un gioco 50/50 fra le due smette di essere un castello da
esplorare e diventa un inseguimento con lo scenario intorno.
`warningDuration` è la manopola dell'equità — accorciarla rende il gioco più
difficile nel modo più economico possibile; se serve più difficoltà, meglio
alzare la `chaseSpeed` dei fantasmi, che almeno il giocatore la vede arrivare.

Il colore della caccia è **viola** `[0.42, 0.10, 1.0]`: letto contro pietra
grigia e luce di fuoco arancione è il segnale "qualcosa non va" più chiaro
disponibile senza aggiungere un elemento di UI. È tutto il motivo per cui la
meccanica è appesa al colore. `huntLightScale = 0.8` abbassa un po' la luce, ma
solo un po': un giocatore che non riesce a leggere i muri non può schivare, e
morire di buio non è la stessa cosa che morire per un fantasma.

## 7.2 I fantasmi

`Ghost` (main.cpp:1782), macchina a tre stati:

- **Patrol** — un giro chiuso di waypoint a velocità costante, con il fantasma rivolto verso dove sta andando.
- **Chase** — durante una caccia: linea di vista, steering, inseguimento.
- **Return** — è lo stato che fa funzionare tutto il resto, e vale la pena spiegarlo. Un fantasma che ha inseguito il giocatore per mezzo dungeon deve tornare alla sua pattuglia, e senza pathfinding non sa come. La soluzione: durante l'inseguimento lascia una **briciola** ogni `GHOST_TRAIL_SPACING` unità, e tornare è semplicemente ripercorrere le briciole a ritroso. Il percorso **si auto-pota**: una briciola nuova che cade vicino a una vecchia taglia via tutto il segmento in mezzo. Un fantasma che ha passato la caccia a girare attorno a un tavolo torna quindi indietro in linea, non ripercorrendo i cerchi.

**Perché i fantasmi usano la technique "Spectral" e non CookTorrance.** All'inizio
erano prop opachi come tutti gli altri, il che dava loro gratis l'ombra nella
mappa 2D del sole e in quelle cubemap delle torce. Poi il sole è stato tolto, e
ri-catturare le cubemap per un occlusore che si muove di continuo è costato più
di quanto valesse l'ombra. Restava un prop grigio opaco senza ombra sotto, e
l'ombra mancante era la cosa più evidente di lui.

`Spectral` risolve rendendolo **incorporeo** invece che opaco: alpha-blended,
unlit, emissivo, con un rim di Fresnel lungo la silhouette. Una cosa attraverso
cui si vede il muro non ha motivo di proiettare ombra, quindi la via economica
smette di sembrare un compromesso. Due conseguenze, entrambe volute: le passate
d'ombra percorrono solo `SC.TI[0]`, quindi i fantasmi ne sono fuori
strutturalmente e non per un controllo `castsShadow`; e non ricevono ombra
nemmeno loro, dato che un emettitore unlit non ha niente da scurire.

**Collisione**: un cilindro verticale, non una box — una box ruoterebbe con lui.
Le due misure sono fittate dalla geometria di `Ghost.gltf` al caricamento. La
fascia verticale è ai bounds esatti della mesh; il raggio invece è
deliberatamente **ridotto** sotto il raggio reale, perché un fantasma è
prevalentemente un drappo e prenderne i bordi come solidi lo farebbe incastrare
in ogni porta.

Lo shrink è 0.45 e non di più per un motivo preciso, che vale come esempio di
compromesso: portandolo a 0.65 il raggio effettivo si avvicinava tanto a metà
larghezza di una porta che il test clear/blocked dello steering cominciava a
cambiare risposta ogni frame, e il fantasma **vibrava** sulla soglia. Un
fantasma che compenetra un tavolo è un difetto visivo minore; uno che vibra in
una porta è rotto. L'obiettivo non è azzerare la compenetrazione, è non
regredire il movimento.

Niente pathfinding: c'è un test contro i muri, un push-out identico a quello del
giocatore, e uno steering che apre a ventaglio da una direzione desiderata a
passi crescenti e prende il primo candidato che non colpisce un muro entro una
certa distanza. Se è chiuso da ogni lato restituisce un vettore nullo, e il
chiamante lo gestisce.

Il ventaglio da solo **ditherebbe**: un fantasma davanti a un pilastro col
giocatore dietro valuterebbe destra e sinistra come equivalenti a ogni frame e,
al muoversi della geometria di centimetri, continuerebbe a scambiarle, vibrando
sul posto invece di impegnarsi. `turnBias` ricorda il lato scelto e lo riprova
per primo, e viene riesaminato solo quando il fantasma ritrova una linea dritta
libera. È lo stesso genere di problema dello shrink a 0.65 di sopra: due
condizioni quasi equivalenti valutate ogni frame vogliono sempre un po' di
memoria o di isteresi.

I fantasmi obbediscono agli stessi muri del giocatore per un motivo di design,
non di realismo: se un fantasma potesse attraversare la parete dietro di te, non
esisterebbe alcun modo di nascondersi.

## 7.3 Porte, oggetti, mira

**Porte** (`Door`, main.cpp:728) — il battente è un'istanza a sé, separata dal
muro in cui è incassato. Si apre ruotando attorno al cardine a
`DOOR_OPEN_SPEED` gradi/secondo. Una porta può essere bloccata e richiedere una
chiave con un certo id. Il raggio di interazione è misurato **in XZ** (non in
3D), alla minore fra la distanza dal vano e quella dal battente.

Dettaglio geometrico che spiega perché il codice è come è: la mesh `SM_Door_01`
ha Z che va da 0.03 a -2.43, cioè tutto il pannello pende da un lato di Z=0.
Bisogna tenerne conto per posizionare cardine e collisione.

**Pickup** (`Pickup`, main.cpp:922) — raccolti con E, raggio misurato in 3D. Il
portachiavi è un `vector` di indici in `pickups` e non un set di id, perché due
chiavi possono condividere lo stesso id (un livello può volere due copie della
stessa chiave). `consumeKey()` cerca **dal fondo**, così la chiave attualmente in
mano è la prima a essere spesa. `G` la rimette a terra, riparcheggiando **la
stessa istanza** che il blocco della mano stava disegnando.

L'animazione di raccolta: la chiave non scatta in mano, sale nella posa finale da
sotto in una frazione di secondo. È un fioretto, non una cutscene — abbastanza
lunga da leggersi, abbastanza corta da non annoiare alla decima chiave.

**La mira (gaze)** — non è un raycast, è un **cono angolare**. `findGazedDoor`,
`findGazedPickup` e `findGazedCandle` valutano l'allineamento fra il forward
normalizzato della camera e la direzione verso l'oggetto, con la semi-larghezza
dell'oggetto che allarga il cono base. Il vincitore diventa `gazedInstance` e
riceve il suo valore di `glow` nell'UBO per-istanza.
`gazedInteractionDisabled` è ciò che rende rossa l'aura.

C'è anche un flag per le porte segrete che il giocatore non deve ancora vedere:
non ricevono l'aura, così l'effetto non le rivela in anticipo.

## 7.4 L'uscita e il finale

La condizione di vittoria è una **box allineata agli assi** definita in
`gameplay.json`, in cui il giocatore deve stare. Il varco d'uscita **non consuma**
la chiave che controlla: la run è finita nell'istante in cui ci si arriva, non ha
senso spendere niente.

La box parte a x 19.6: dentro l'apertura, ma oltre il piano in cui sta il
battente chiuso, quindi nessuno può trovarcisi dentro finché la porta non è
stata sbloccata e aperta. E non va oltre, deliberatamente: la luce del giorno sta
a x 22.0 con la porta che si apre davanti, e una box più esterna consegnerebbe la
vittoria al giocatore mentre guarda il *retro* dell'effetto. Il premio deve
arrivare mentre l'abbaglio riempie ancora lo schermo.

**ExitGlow** — la luce del giorno fuori dalla porta, tre quad (shader in §3.13).
Perché tre e non uno è pura geometria: il muro è spesso (x 18.758..20) e l'arco è
profondo (z 28.87..31.11), quindi da certi angoli si guarda oltre il piano del
primo quad e si vedrebbe il bordo; il secondo copre quel caso, il terzo è quello
a terra capovolto, che copre la linea di vista attraverso la parte alta
dell'arco. Nessun valore di `EXIT_GLOW_HALF_HEIGHT` può risolvere quel caso, per
questo serve un quad in più e non un quad più grande. Tutti e tre sopravanzano
generosamente ciò che devono coprire, con entrambe le estremità sepolte nella
geometria.

In più, il varco getta una **spot light all'indietro** dentro la stanza,
aggiunta alla lista delle luci: è un'apertura sul giorno pieno dentro una stanza
illuminata a torce, quindi ha un valore ben sopra 1. Il cono è più stretto di
quanto sembrerebbe naturale, perché questa luce non ha ombre proprie e un cono
largo la farebbe passare attraverso i muri.

**Il whiteout.** `escapeFlash` va da 0 a 1 in `PostUniformBufferObject`, sopra la
rampa di esposizione e bloom che gira nello stesso momento, e viene applicato
**dopo** il tone map (§3.11). L'ordine è ciò che dà la sensazione giusta: la
scena prima esplode di luce, fa bloom e perde dettaglio, e solo dopo l'ultimo di
essa si lava via. Solo la seconda metà sarebbe un fade a bianco, cioè una
transizione di schermo, non l'essere accecati.

Deve essere un termine a sé e non semplicemente più esposizione perché il tone
map `c/(Y+1)` tende al bianco asintoticamente: nessuna esposizione, per quanto
grande, arriva davvero al bianco, e peggio, ci arriva a velocità molto diverse
per un muro illuminato e per un angolo buio. Alzare solo l'esposizione non
sbianca il frame, lo appiattisce in un grigio con le parti brillanti ancora in
vantaggio.

## 7.5 Camera, collisioni, movimento

Camera free-look: `camPos`, `camYaw` (0 guarda verso +X, crescente gira a
destra), `camPitch` (-90 giù, +90 su). Spawn dentro la sala del dungeon a
`(-33.5, 1.8, 29.0)`, girata verso +X così si guarda dritti lungo il corridoio
verso la porta in fondo, libera dal tavolo e da entrambe le torce.

Gravità: `camVerticalVelocity`, azzerata ogni volta che il clamp col terreno
scatta (cioè siamo atterrati). Il salto ha edge detection, così tenere premuto
non ri-triggera. Lo sprint (Ctrl) si può **iniziare** solo da terra — niente
sprint partito a mezz'aria — ma si può **fermare** in aria.

**Collisioni**: `Colliders.hpp` del framework costruisce AABB automatiche per i
modelli che le dichiarano in `scene.json` con `"collider": "AABB"`;
`SceneColliders.hpp` carica `colliders.json` con la geometria autorata a mano per
i casi in cui una box auto-fittata sbaglia — tipicamente l'arcata di un varco,
dove serve un buco al centro e una box piena lo tapperebbe. I due insiemi
vengono uniti in `allColliders`, tenuto come membro suo così i loop di collisione
per-frame leggono un vector semplice invece di passare dall'accessor.

<mark>`MAX_STEP_HEIGHT` è la superficie più alta su cui si sale camminando senza
saltare. `worldFloorY` è l'altezza del pavimento, cacheata una volta, usata come
clamp di ultima istanza in no-clip così cadere sotto la mappa è impossibile.</mark>
Sono i due numeri che rendono il movimento tollerabile: il primo evita di dover
saltare su ogni gradino e su ogni bordo di tappeto, il secondo è una rete di
sicurezza per quando il cheat delle collisioni è spento e non c'è più niente a
tenerti sopra la geometria.

**Smoothing verticale della vista**: il clamp col terreno alza la camera
istantaneamente nell'istante in cui i piedi toccano qualcosa di più alto, e uno
scatto istantaneo della camera è nauseante. Questo decadimento assorbe lo scatto
con una costante di tempo, con un tetto sul singolo scatto che è disposto ad
assorbire (oltre quel tetto è meglio far vedere il salto che far scivolare la
vista per mezzo secondo). Le rampe autorate sono già smooth e lo alimentano
appena; serve soprattutto ai gradini.

**Walk bob**: un'oscillazione laterale una volta per passo più un rimbalzo
verticale, con `walkBobBlend` che ne fa il fade in e out invece di accenderlo di
colpo. La fase avanza più in fretta se si sprinta. Alimenta anche il `lean` della
fiamma, così la torcia risponde al passo.

> **Se il prof chiede**
>
> *"Come fanno i fantasmi a tornare alla pattuglia dopo un inseguimento?"* — Con
> una scia di briciole lasciata durante l'inseguimento e ripercorsa a ritroso,
> auto-potata quando una briciola nuova cade vicino a una vecchia. Non è
> pathfinding.
>
> *"Il whiteout finale è un fade a bianco?"* — No, ed è la differenza che conta.
> È applicato dopo il tone map, sopra una rampa di esposizione e bloom, quindi la
> scena prima si sovraespone e perde dettaglio e solo dopo si lava via. Un fade
> è una transizione di schermo, questo è l'essere accecati.

---

# PARTE 8 — I file di dati

Tutti in `assets/scenes/`, tutti con commenti (`json.hpp` in modalità
permissiva). Il senso comune a tutti: cambiare un numero cambia il gioco senza
ricompilare, e quel file esiste perché quel numero **si tara guardando il
risultato**, non ragionandoci.

- `scene.json` — modelli, texture, istanze, tecniche. Parsato da `Scene.hpp`, che legge solo id/model/texture/translate/eulerAngles/scale. È il motivo per cui esistono tutti gli altri.
- `materials.json` — un `Material` per **modello** (non per istanza, così le quattro torri condividono una voce invece di ripeterla 23 volte): `specularColor`, `roughness`, `F0`, `k`, più i flag `flatNormals`, `interiorAmbient`, `metallic`, `castsShadow`, e l'override `ambientWeight`.
- `lights.json` — le sorgenti e l'ambient emisferico.
- `flames.json` — quali modelli ricevono una fiamma, con quale ancora, scala, colore.
- `colliders.json` — geometria di collisione autorata a mano, per istanza.
- `gameplay.json` — tempi del ciclo di caccia, pattuglie dei fantasmi, box di vittoria, chiave dell'uscita.

Su `materials.json` vale la pena sapere che significano i quattro parametri, se
te lo chiedono:

- `specularColor` — il colore dell'highlight. Bianco per tutto ciò che non è metallo, perché l'highlight ha il colore della **lampada**, non dell'oggetto. Solo i metalli lo tingono.
- `roughness` — 0 è uno specchio con un highlight minuscolo e netto, 1 è gesso senza highlight. È quello che si tara davvero.
- `F0` — riflettanza guardando in faccia. Circa 0.04 per ogni non-metallo esistente, quindi si copia più che sceglierlo. I metalli sono molto più alti.
- `k` — quanta della risposta è colore normale contro highlight. Alto per i materiali ordinari, ignorato per quelli con `metallic`, che non hanno diffuso.

Una sola tecnica è registrata: `PRs[0] = "CookTorrance"`, con i suoi tre
descriptor set. Fiamme, ExitGlow, UiQuad e LightDebug **non** passano da `Scene`:
hanno pipeline proprie e si registrano da sole nel command buffer.

---

# PARTE 9 — Menu cheat e viste di debug

`L` apre il pannello (`CheatHud.hpp`), navigabile con frecce + Invio oppure col
mouse. Ogni riga tiene un `bool*` vivo a un membro di `CheatFlags` in `main.cpp`:
la HUD non copia niente e non possiede niente, capovolge il flag reale sul posto,
quindi non c'è nulla da tenere in sync.

Il pannello si ridisegna solo quando qualcosa cambia (apertura, selezione
spostata, toggle premuto), non a ogni frame in cui resta aperto e fermo — stessa
logica per cui il contatore FPS chiama `txt.print` una volta al secondo e non a
ogni frame.

Il testo viene da `TextMaker`, lo sfondo e l'evidenziazione da `UiQuad`, perché
l'atlante del font non ha un rettangolo pieno con cui simulare un pannello.

Le viste di debug del modello di luce sono bit in `gubo.debugFlags`
(definiti in `LightConstants.glsl`). Sono un **bitmask** e non quattro int
separati perché sono interruttori indipendenti che si combinano, e quattro int
mangerebbero quattro slot di uniform. Sono anche **uniform branch**: in un frame
normale sono tutti spenti, quindi ogni pixel di ogni draw prende lo stesso ramo,
che sulla GPU è il caso economico.

- `LIGHT_DEBUG_UNLIT` (1) — solo albedo, nessuna illuminazione e nessun ambient. Distingue "nessuna luce è arrivata qui" da "il texel della texture è nero".
- `LIGHT_DEBUG_NORMALS` (2) — la normale di shading come colore, rimappata da [-1,1] a [0,1], quindi +X rosso, +Y verde, +Z blu. Piazzata **dopo** il blocco `flatNormals`, così mostra la normale che l'illuminazione ha davvero usato, faccette incluse.
- `LIGHT_DEBUG_NO_SPECULAR` (4) — forza `k = 1`, quindi il termine speculare resta moltiplicato per 0: spariscono gli highlight, tutto il resto resta com'era.
- `LIGHT_DEBUG_NO_TONEMAP` (8) — letto da `Composite.frag`, non più da `CookTorrance.frag`. Il cheat funziona ancora, semplicemente ha effetto un pass più tardi.
- `LIGHT_DEBUG_NO_SHADOWS` (16) — ogni `shadowFactor()` forzato a 1.
- `LIGHT_DEBUG_HEATMAP` (32) — ricolora per intensità di luce in arrivo, ignorando l'albedo. C'è una compressione logaritmica prima della rampa, perché l'ingresso è radianza HDR non clampata: una rampa lineare mostrerebbe rosso pieno su quasi tutta una stanza illuminata invece del decadimento che il cheat esiste per visualizzare.

Accendere e spegnere **intere categorie** di luce non sta qui: succede lato CPU
in `SceneLights`, che semplicemente non le carica.

Altri toggle: collisioni (no-clip), ombre delle torce, ombre delle candele,
overlay delle posizioni delle luci (`LightDebug.hpp`), stampa continua della
posizione della camera.

---

# PARTE 10 — Controlli

- `WASD` — movimento. `R` / `F` su e giù, `Q` / `E` e le frecce per la rotazione (dal `getSixAxis` del framework).
- Mouse — sguardo.
- `Spazio` (o tasto destro del mouse) — salto, con edge detection così tenerlo premuto non ri-triggera.
- `Ctrl` — sprint (×2, avviabile solo da terra).
- `E` — interagisci: apri o chiudi una porta, raccogli un oggetto, accendi una candela.
- `G` — posa a terra la chiave che hai in mano.
- `R` — riavvia la run. `restartRun()` rimette tutto allo stato autorato **senza rileggere nessun file**: ogni valore "autorato" era stato catturato in `localInit()`.
- `L` — menu cheat. Frecce + Invio dentro.
- `Esc` — esci.

---

# PARTE 11 — Numeri chiave, tutti in un posto

- `MAX_LIGHTS` = 32, `NUM_SHADOW_MAPS_2D` = 2, `NUM_SHADOW_CUBES` = 32, `SHADOW_CUBE_RES` = 1024.
- MSAA 4×, con per-sample shading forzato da `Starter.hpp`.
- Bloom: un quarto di risoluzione (`BLOOM_DIV` = 4), soglia 1.55, knee 0.45, intensità 0.65, esposizione 1.0.
- Cube shadow: near 0.05, far 60.0 (che è anche il clear value), bias 0.02..0.06, normal offset ≤ 0.12.
- Riassegnazione del pool dinamico ogni 0.3 s, margine di swap 1.15.
- Ciclo di caccia: 45 / 3 / 18 s, recupero 4 s. Colore della caccia viola `[0.42, 0.10, 1.0]`, scala luce 0.8.
- HDR boost: fiamma fino a 6×, scintille 12× → 3×, ExitGlow molto oltre.
- Inviluppi della fiamma: luminosità 0.30..1.40, altezza 0.78..1.09.

**Un vincolo di cui essere consapevoli**, perché è il genere di cosa su cui un
prof può incalzare: 32 `samplerCube` + 2 `sampler2D` + 1 albedo = 35 immagini
campionate nello stage fragment, ben oltre il minimo **garantito** da Vulkan
(`maxPerStageDescriptorSampledImages` = 16). È una scelta deliberata — le GPU
desktop reali ne permettono molte di più, e `flames.json` fa sì che il numero di
point light dipenda dal livello e non sia più fisso — ma va detto: il progetto
non gira su una GPU al minimo di specifica. Se dovesse servire, il pool dinamico
degrada in modo pulito: `SHADOW_SWAP_MARGIN` avrebbe semplicemente più candidati
da arbitrare fra meno slot.
