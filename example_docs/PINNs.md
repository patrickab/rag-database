---
aliases: [PINNs, Physik-informierte Neuronale Netze, Physics-Informed Machine Learning]
tags: [concept, machine-learning, scientific-computing, differential-equations, neural-networks, deep-learning]
summary: "PINNs sind hybride Modelle, die physikalische Gesetze (formuliert als PDEs) direkt in die Verlustfunktion neuronaler Netze integrieren, um komplexe physikalische Systeme ohne traditionelle Gitter zu lösen."
---

# Physik-Informierte Neuronale Netze (PINNs)

Stellen Sie sich vor, Sie möchten ein komplexes physikalisches Phänomen wie den Luftstrom um ein Flugzeug oder die Ausbreitung eines Virus modellieren. Traditionell würden Sie auf numerische Löser (wie die Finite-Elemente-Methode) zurückgreifen, die aufwendige Gittererstellung (Meshing) und immense Rechenleistung erfordern. Auf der anderen Seite könnten Sie ein rein datengetriebenes neuronales Netz verwenden, das jedoch Unmengen an Trainingsdaten benötigt und keine Garantie dafür bietet, grundlegende physikalische Gesetze wie die Energieerhaltung zu respektieren.

PINNs schlagen eine elegante Brücke zwischen diesen beiden Welten. Sie nutzen die universelle Approximationsfähigkeit neuronaler Netze, um die Lösung einer Differentialgleichung zu lernen, zwingen das Netz aber während des Trainings, die physikalischen Gesetze selbst zu erfüllen. Das "Informieren" geschieht, indem das Residuum der Differentialgleichung Teil der Verlustfunktion wird. Ein PINN lernt also nicht nur aus Datenpunkten, sondern auch direkt aus der zugrundeliegenden Physik.

## Inhaltsverzeichnis
- [Das große Ganze: Die Kernidee von PINNs](#das-große-ganze-die-kernidee-von-pinns)
- [Erfolgsgeschichten: PINNs in der Praxis](#erfolgsgeschichten-pinns-in-der-praxis)
- [Die grundlegende Architektur eines PINN](#die-grundlegende-architektur-eines-pinn)
    - [Das neuronale Netz als Lösungsansatz](#das-neuronale-netz-als-lösungsansatz)
    - [Die Verlustfunktion: Das Herzstück des PINN](#die-verlustfunktion-das-herzstück-des-pinn)
- [Populäre Erweiterungen und Varianten](#populäre-erweiterungen-und-varianten)
    - [cPINNs (Conservative PINNs)](#cpinns-conservative-pinns)
    - [fPINNs (Fractional PINNs)](#fpinns-fractional-pinns)
    - [XPINNs (Extended PINNs)](#xpinns-extended-pinns)
- [Die Mathematik hinter PINNs: Eine intuitive Herleitung](#die-mathematik-hinter-pinns-eine-intuitive-herleitung)
    - [Problemformulierung: Die PDE](#problemformulierung-die-pde)
    - [Der Lösungsansatz: Das neuronale Netz](#der-lösungsansatz-das-neuronale-netz)
    - [Die Magie der Verlustfunktion](#die-magie-der-verlustfunktion)
- [Die entscheidenden Vorteile von PINNs](#die-entscheidenden-vorteile-von-pinns)
- [Reflexion und Lernziele](#reflexion-und-lernziele)

## Das große Ganze: Die Kernidee von PINNs

Bevor wir in die Details eintauchen, lassen Sie uns die konzeptionelle Architektur skizzieren. Ein PINN besteht aus zwei fundamentalen Komponenten:

1.  **Ein universeller Funktionsapproximator**: In der Regel ein einfaches, vollständig verbundenes neuronales Netz (MLP). Dieses Netz, nennen wir es $\hat{u}(x, t; \theta)$, nimmt als Input Koordinaten (z.B. Ort $x$ und Zeit $t$) und gibt einen Schätzwert für die physikalische Größe $u$ aus. Die Parameter $\theta$ sind die Gewichte und Biases des Netzes.

2.  **Eine physikalisch informierte Verlustfunktion**: Dies ist die entscheidende Innovation. Die Verlustfunktion $L(\theta)$ besteht aus mehreren Termen:
    *   **Daten-Verlust ($L_{data}$)**: Ein klassischer Term (z.B. Mean Squared Error), der die Abweichung der Netzvorhersage von bekannten Messpunkten, Anfangs- oder Randbedingungen misst.
    *   **Physik-Verlust ($L_{phys}$)**: Dieser Term misst, wie gut die vom Netz angenäherte Lösung $\hat{u}$ die zugrundeliegende partielle Differentialgleichung (PDE) erfüllt. Um dies zu berechnen, werden die Ableitungen von $\hat{u}$ benötigt. Der Clou hierbei ist der Einsatz von **Automatischer Differentiation (AD)**, einer Technik, die in allen modernen Deep-Learning-Frameworks (wie PyTorch oder TensorFlow) implementiert ist. AD erlaubt es uns, die exakten Ableitungen des Netzausgangs nach seinen Eingängen zu berechnen, ohne auf numerische Approximationen zurückgreifen zu müssen.

Das Training eines PINN ist dann ein Optimierungsproblem: Finde die Netzwerkparameter $\theta$, die die kombinierte Verlustfunktion $L(\theta) = \lambda_{data} L_{data} + \lambda_{phys} L_{phys}$ minimieren. Das Netz lernt also gleichzeitig, die Datenpunkte zu treffen *und* die physikalischen Gesetze im gesamten Definitionsbereich zu befolgen.

## Erfolgsgeschichten: PINNs in der Praxis

Die Eleganz dieses Ansatzes hat zu beeindruckenden Erfolgen in verschiedensten Domänen geführt:

- **Fluiddynamik**: Simulation von komplexen Strömungen, wie der Navier-Stokes-Gleichungen, ohne die Notwendigkeit eines Rechengitters (mesh-free). Dies ist besonders vorteilhaft bei komplexen Geometrien, wo die Gittererzeugung oft den größten Aufwand darstellt.
- **Biomedizinische Technik**: Modellierung von Blutfluss in Aneurysmen oder Tumorwachstum. Hier können PINNs spärliche, nicht-invasive Messdaten (z.B. aus MRT-Scans) mit biomechanischen Modellen kombinieren, um personalisierte Vorhersagen zu treffen.
- **Materialwissenschaft**: Lösung sogenannter *inverser Probleme*. Beispiel: Aus der Beobachtung der Verformung eines Materials unter Last ($\rightarrow$ Daten) können PINNs auf die unbekannten Materialparameter (z.B. Elastizitätsmodul) schließen, indem diese Parameter als trainierbare Variablen in das Modell aufgenommen werden.
- **Quantenmechanik**: Lösung der hochdimensionalen Schrödinger-Gleichung, bei der traditionelle gitterbasierte Methoden an der "Fluch der Dimensionalität" (Curse of Dimensionality) scheitern.

## Die grundlegende Architektur eines PINN

#### Das neuronale Netz als Lösungsansatz

Das Herzstück ist ein neuronales Netz, das die gesuchte Lösungsfunktion $u(x, t)$ approximiert.

- **Input**: Die unabhängigen Variablen des Problems, typischerweise Raum- und Zeitkoordinaten $(x, y, z, t)$.
- **Architektur**: Meist ein Multi-Layer Perceptron (MLP) mit mehreren versteckten Schichten und Aktivierungsfunktionen wie $\tanh$ oder $\sin$, da deren Ableitungen glatt und nicht-null sind, was für die Berechnung der PDE-Terme wichtig ist.
- **Output**: Die abhängigen Variablen, also die physikalischen Felder, die durch die PDE beschrieben werden (z.B. Geschwindigkeit $v$, Druck $p$, Temperatur $T$).

Wir bezeichnen die Approximation des Netzes als $\hat{u}(x, t; \theta)$, wobei $\theta$ die Menge aller trainierbaren Gewichte und Biases darstellt.

#### Die Verlustfunktion: Das Herzstück des PINN

Die Gesamtverlustfunktion $L(\theta)$ ist eine gewichtete Summe aus zwei Hauptkomponenten:

1.  **$L_{data}(\theta)$**: Der Verlust an den Datenpunkten.
    - Dies umfasst Anfangsbedingungen (IC), Randbedingungen (BC) und alle sonstigen verfügbaren Messdaten.
    - Typischerweise wird der mittlere quadratische Fehler (MSE) verwendet:
    $$
    L_{data}(\theta) = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} |\hat{u}(x_i, t_i; \theta) - u_i|^2
    $$
    wobei $(x_i, t_i)$ die Koordinaten der Datenpunkte und $u_i$ die zugehörigen Messwerte sind.

2.  **$L_{phys}(\theta)$**: Der Physik-Verlust oder Residuum-Verlust.
    - Sei die PDE gegeben durch $f(u, \frac{\partial u}{\partial t}, \frac{\partial u}{\partial x}, ...) = 0$.
    - Das Residuum des Netzes ist definiert als $r(x, t; \theta) = f(\hat{u}, \frac{\partial \hat{u}}{\partial t}, \frac{\partial \hat{u}}{\partial x}, ...)$.
    - Die Ableitungen wie $\frac{\partial \hat{u}}{\partial t}$ werden mittels **Automatischer Differentiation** direkt aus dem Graphen des neuronalen Netzes berechnet.
    - Der Physik-Verlust minimiert das Residuum an einer großen Anzahl von zufällig im Raum-Zeit-Gebiet gewählten Punkten, den sogenannten **Kollokationspunkten**.
    $$
    L_{phys}(\theta) = \frac{1}{N_{coll}} \sum_{j=1}^{N_{coll}} |r(x_j, t_j; \theta)|^2
    $$
    ⚠️ **Wichtige Einsicht**: Das Netz wird nicht nur dort korrigiert, wo wir Daten haben, sondern *überall* im Definitionsbereich, indem es gezwungen wird, die physikalischen Gesetze zu befolgen. Dies wirkt als extrem starker Regularisierer und ermöglicht das Lernen aus sehr wenigen Datenpunkten.

## Populäre Erweiterungen und Varianten

Das Grundkonzept der PINNs ist sehr flexibel und hat zu einer Vielzahl von Erweiterungen geführt:

#### cPINNs (Conservative PINNs)
- **Problem**: Standard-PINNs garantieren nicht die Einhaltung von Erhaltungssätzen (z.B. Masse, Impuls, Energie), die oft in integraler Form vorliegen. Kleine lokale Fehler im Residuum können sich zu signifikanten globalen Fehlern in den Erhaltungsgrößen aufsummieren.
- **Lösung**: cPINNs modifizieren die Architektur oder die Verlustfunktion, um diese Erhaltungssätze explizit zu erzwingen. Ein Ansatz ist, die PDE in ihrer Divergenzform zu formulieren und dies in der Netzarchitektur abzubilden.

#### fPINNs (Fractional PINNs)
- **Problem**: Viele komplexe Phänomene in der Physik und im Finanzwesen werden durch fraktionale PDEs beschrieben, die Ableitungen nicht-ganzzahliger Ordnung beinhalten.
- **Lösung**: Die Flexibilität der automatischen Differentiation kann erweitert werden, um auch fraktionale Ableitungen des Netzes zu berechnen, was die Anwendung von PINNs auf diese exotischere Klasse von Problemen ermöglicht.

#### XPINNs (Extended PINNs)
- **Problem**: Das Training eines einzigen großen PINNs für sehr große oder komplexe Domänen kann schwierig sein (z.B. aufgrund von spektralem Bias oder verschwindenden Gradienten).
- **Lösung**: XPINNs verwenden einen Domain-Decomposition-Ansatz. Die Gesamtdomäne wird in kleinere Subdomänen aufgeteilt, und für jede Subdomäne wird ein eigenes kleines PINN trainiert. Die Kontinuität der Lösung und ihrer Ableitungen an den Schnittstellen der Subdomänen wird durch zusätzliche Terme in der Verlustfunktion sichergestellt.

## Die Mathematik hinter PINNs: Eine intuitive Herleitung

#### Problemformulierung: Die PDE

Betrachten wir eine allgemeine, nichtlineare PDE der Form:
$$
\frac{\partial u}{\partial t} + \mathcal{N}[u] = 0, \quad x \in \Omega, \quad t \in [0, T]
$$
mit Randbedingungen (BC) $\mathcal{B}(u, x, t) = 0$ auf $\partial\Omega$ und Anfangsbedingungen (IC) $u(x, 0) = g(x)$. Hier ist $\mathcal{N}[\cdot]$ ein nichtlinearer Differentialoperator.

Wir definieren das Residuum der PDE als:
$$
f(x, t) := \frac{\partial u}{\partial t} + \mathcal{N}[u]
$$
Das Ziel ist es, eine Funktion $u(x, t)$ zu finden, für die $f(x, t) = 0$ im gesamten Definitionsbereich $\Omega \times [0, T]$ gilt und die die IC/BC erfüllt.

#### Der Lösungsansatz: Das neuronale Netz

Wir approximieren die Lösung $u(x, t)$ durch ein neuronales Netz $\hat{u}(x, t; \theta)$.

#### Die Magie der Verlustfunktion

Die kombinierte Verlustfunktion $L(\theta)$ wird aus drei Teilen zusammengesetzt:

1.  **Verlust der Anfangsbedingung ($L_{IC}$)**:
    $$
    L_{IC}(\theta) = \frac{1}{N_{IC}} \sum_{i=1}^{N_{IC}} |\hat{u}(x_i, 0; \theta) - g(x_i)|^2
    $$
    Hier sind $\{x_i\}_{i=1}^{N_{IC}}$ Punkte aus dem räumlichen Gebiet $\Omega$ zur Zeit $t=0$.

2.  **Verlust der Randbedingung ($L_{BC}$)**:
    $$
    L_{BC}(\theta) = \frac{1}{N_{BC}} \sum_{j=1}^{N_{BC}} |\mathcal{B}(\hat{u}, x_j, t_j; \theta)|^2
    $$
    Hier sind $\{(x_j, t_j)\}_{j=1}^{N_{BC}}$ Punkte auf dem Rand $\partial\Omega$ der Domäne.

3.  **Verlust des PDE-Residuums ($L_{phys}$)**:
    $$
    L_{phys}(\theta) = \frac{1}{N_{coll}} \sum_{k=1}^{N_{coll}} |f(x_k, t_k; \theta)|^2
    $$
    wobei $f(x_k, t_k; \theta) := \frac{\partial \hat{u}}{\partial t}(x_k, t_k; \theta) + \mathcal{N}[\hat{u}(x_k, t_k; \theta)]$. Die Punkte $\{(x_k, t_k)\}_{k=1}^{N_{coll}}$ sind die Kollokationspunkte, die im Inneren der Domäne verteilt sind.

Die Gesamtverlustfunktion ist dann:
$$
L(\theta) = \lambda_{IC} L_{IC}(\theta) + \lambda_{BC} L_{BC}(\theta) + \lambda_{phys} L_{phys}(\theta)
$$
Die Hyperparameter $\lambda_{IC}, \lambda_{BC}, \lambda_{phys}$ gewichten die einzelnen Terme und ihre Wahl ist entscheidend für den Trainingserfolg.

## Die entscheidenden Vorteile von PINNs

✅ **Gitterfrei (Mesh-free)**: PINNs benötigen kein explizites Rechengitter. Die Physik wird an beliebigen Kollokationspunkten erzwungen. Dies ist ein enormer Vorteil bei Problemen mit komplexen Geometrien oder sich bewegenden Rändern.

✅ **Hybrid aus Daten und Physik**: Sie können nahtlos spärliche, verrauschte Messdaten mit physikalischem Wissen kombinieren. Das Modell interpoliert zwischen den Datenpunkten auf eine physikalisch plausible Weise.

✅ **Lösung von inversen Problemen**: Einer der stärksten Anwendungsfälle. Unbekannte Parameter in der PDE (z.B. Viskosität, Wärmeleitfähigkeit) können einfach als trainierbare Variablen zum Netzwerk hinzugefügt werden. Das PINN findet dann gleichzeitig die Lösung *und* die Parameter, die am besten zu den Beobachtungsdaten passen.

✅ **Potenzial für hochdimensionale Probleme**: Während gitterbasierte Methoden exponentiell mit der Anzahl der Dimensionen skalieren (Curse of Dimensionality), ist die Komplexität von PINNs (definiert durch die Anzahl der Kollokationspunkte) davon weniger stark betroffen. Dies eröffnet Möglichkeiten zur Lösung von Problemen wie der Black-Scholes-Gleichung in der Finanzmathematik oder der Schrödinger-Gleichung.

## Reflexion und Lernziele

💡 **Zentrale Erkenntnisse**:
- PINNs sind keine reinen Black-Box-Modelle; sie integrieren Domänenwissen in Form von Differentialgleichungen direkt in den Lernprozess.
- Die Magie liegt in der Kombination eines universellen Funktionsapproximators (NN) mit Automatischer Differentiation, um eine physikalisch informierte Verlustfunktion zu konstruieren.
- Sie verschieben das Problem von der Lösung eines komplexen Gleichungssystems auf einem Gitter hin zu einem hochdimensionalen Optimierungsproblem im Parameterraum des neuronalen Netzes.
- Ihre Stärke liegt insbesondere in der Lösung inverser Probleme und der Arbeit mit spärlichen Daten, wo traditionelle Methoden oft versagen.

🎯 **Lernziele**:
- [ ] Erklären Sie die konzeptionelle Idee hinter PINNs und wie sie sich von rein datengetriebenen und traditionellen numerischen Methoden unterscheiden.
- [ ] Skizzieren Sie die Architektur eines PINN und die Rolle der einzelnen Komponenten (NN, Verlustfunktion, Kollokationspunkte).
- [ ] Formulieren Sie die zusammengesetzte Verlustfunktion für ein gegebenes PDE-Problem (z.B. die Wärmeleitungsgleichung) mit Anfangs- und Randbedingungen.
- [ ] Erläutern Sie die entscheidende Rolle der Automatischen Differentiation im Kontext von PINNs.
- [ ] Vergleichen Sie die Vor- und Nachteile von PINNs gegenüber klassischen Lösungsverfahren wie der Finiten-Elemente-Methode (FEM).
- [ ] Identifizieren Sie Problemklassen (z.B. inverse Probleme), für die PINNs besonders gut geeignet sind.