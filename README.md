# Parallax 3D — Python

## Versão Python

```sh
./run.sh
# Ou iniciar em tela cheia:
./run.sh --fullscreen
```

O cubo colorido fica parado, sem textura de dado. A posição do rosto controla a órbita da câmera; subir a cabeça revela a face superior e abaixar revela a inferior. O tamanho do rosto controla a distância. Virar o rosto sem deslocá-lo não é medido como rotação. O controle usa suavização leve (`EMA = 0.35`), sem calibração automática ou zona morta. A janela exibe a cena e a prévia espelhada da webcam no canto, sem textos. **F** alterna tela cheia; **Q/Esc** sai. Feche outros aplicativos que estejam usando a webcam antes de executar.

- `paralax_3d.py`: execução, webcam e teclado.
- `tracking.py`: YuNet e fallback Haar.
- `controls.py`: posição, suavização e sensibilidade.
- `scene.py`: geometria, projeção e renderização.
- `tests/test_python.py`: testes Python.

Testes e imagens de referência: `./run.sh --test` (não usa a webcam).

A geometria fixa, as normais e as cores são pré-calculadas. A projeção e o recorte das linhas são feitos em lote com NumPy; cada aresta visível é desenhada uma única vez. A sensibilidade vertical fica em `GAIN_Y`, em `controls.py`.

Para recriar o ambiente: `python3 -m venv .venv`, depois `.venv/bin/pip install -r requirements.txt`.

## Cena da parede — espiar com a cabeça

```sh
./run.sh --scene parede
# Opcional:
./run.sh --scene parede --fullscreen
```

Uma parede 3D lisa, menor e recuada, ocupa parte da metade esquerda da tela na posição central. Ela tem espessura: ao deslocar a cabeça para a direita, você vê sua lateral e espia os objetos escondidos atrás dela. Ao voltar para a esquerda, os objetos ficam escondidos. A câmera virtual desliza com a cabeça, sem orbitar ou girar os objetos. As faces, a borda e o fundo são projetados em perspectiva e se deslocam conforme a profundidade. É preciso deslocar o rosto dentro do quadro, não apenas girar o pescoço.

A cena original continua em `./run.sh` ou `./run.sh --scene cubo`. O arquivo `wall_scene.py` contém a nova cena; os controles de tela cheia e a prévia da webcam são os mesmos. `./run.sh --test` também salva vistas de referência da parede.

A parede tem uma janela aberta no centro, sem vidro, com profundidade nas bordas. Você pode espiar por ela ou pela lateral da parede; o fundo continua visível através da abertura.
