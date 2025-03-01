---
title: Explorando a Mescla de TD-Learning e Monte Carlo

---

# Explorando a Mescla de TD-Learning e Monte Carlo

Gabriel Santos e silva, Ronaldo Rodrigues e Vicente Sampaio 

## Introdução

Métodos de **Aprendizado por Reforço** (Reinforcement Learning, RL) serão, para o experimento que se segue, expressos da seguinte forma:  
- **Monte Carlo (MC):** Usa o retorno total ao final de cada episódio para atualizar as estimativas de valor, o que elimina o viés de bootstrapping mas pode ter alta variância.  
- **Temporal-Difference (TD):** Atualiza valores a cada passo, usando a estimativa do estado seguinte (bootstrapping). Tende a ter menos variância, mas pode introduzir algum viés.

O objetivo deste projeto é **investigar uma abordagem híbrida**, em que realizamos atualizações do tipo TD (Sarsa) a cada passo e, ao final do episódio, ajustamos nossas estimativas usando o retorno Monte Carlo. Dessa forma, buscamos equilibrar viés e variância.  

Este artigo documenta nossos experimentos iniciais nos ambientes **CliffWalking** **FrozenLake** e **Taxi** do Gymnasium.

---

## Ambientes:

### CliffWalking-v2 

O *CliffWalking* é um ambiente clássico do Gymnasium, em que o agente começa em um dos cantos de um ambiente e precisa chegar ao outro canto sem cair do “penhasco” (as células na borda). As recompensas são definidas como:

- `-1` por passo (custo de movimento);
- `-100` ao cair no penhasco, retornando ao estado inicial.

O episódio termina quando o agente chega ao estado terminal ou (neste experimento) quando atinge **500 passos** (TimeLimit).

### Frozen Lake
O ambiente Frozen Lake do Gymnasium é um ambientede aprendizado por reforço onde um agente precisa navegar de um ponto inicial até um objetivo em um lago congelado sem cair em buracos. O ambiente é representado por uma grade quadrada com cada espaço representando um estado diferente, com o agente podendo mover-se em 4 direções.
As recompensas são definidas como: 

- `+1` ao alcançar o objetivo.
- `0` para qualquer outro movimento.
Perigos: Algumas células possuem buracos onde o agente cai e perde o jogo.

### Taxi V3
O ambiente Taxi-v3 do Gymnasium é um problema onde um táxi precisa pegar e deixar passageiros no local correto dentro de uma cidade representada como um grid.

- O ambiente possui um Grid 5x5: O mapa contém quatro locais marcados (R, G, Y, B) onde os passageiros podem ser pegos e deixados. 

As recompensas são definidas como:
- `+20` ao deixar o passageiro no destino correto.
- `1` por cada ação para incentivar eficiência.
- `10` ao tentar pegar ou soltar o passageiro no lugar errado.


---

## Métodos Utilizados

### 1. Monte Carlo (MC)
- Executamos episódios completos, armazenamos as transições `(estado, ação, recompensa)` e, ao final de cada episódio, calculamos o **retorno descontado** para cada par `(estado, ação)`.  
- Atualizamos `Q(estado, ação)` usando a **média** dos retornos observados.  

### 2. SARSA (TD de 1 passo)
- Atualização a cada passo:  
  $[
  Q(s,a) \gets Q(s,a) + \alpha \big[r + \gamma\,Q(s',a') - Q(s,a)\big]]$ 
- Escolha da ação via **epsilon-greedy**.  
- É um método **on-policy**, pois atualiza a política que está sendo seguida.

### 3. TD + MC Mix
- Durante o episódio, fazemos atualizações TD (semelhantes ao SARSA).  
- Ao final do episódio, recalculamos o **retorno total** (Monte Carlo) para cada transição e ajustamos `Q(s,a)` como uma mescla:  
  $[
  Q_{\text{novo}}(s,a) = w_{\text{MC}} \times Q_{\text{MC}}(s,a) \;+\; (1 - w_{\text{MC}})\times Q_{\text{TD}}(s,a)
  ]$ 
- A ideia é “corrigir” ou “refinar” o valor aprendido por TD com a estimativa de Monte Carlo.

---

## Principais Configurações

- **Gym Environment:** `CliffWalking-v0` com `TimeLimit(env, max_episode_steps=500)`.
- **Número de Episódios:** 1000 (em alguns testes, variamos para menos ou mais).
- **Parâmetros:**  
  - `alpha = 0.1` (para métodos TD)  
  - `epsilon = 0.1` (exploração)  
  - `gamma = 1.0` (fator de desconto; ajustado de 1.5 para 1.0 ao longo dos experimentos)  
  - `mc_weight = 0.5` (no método híbrido)

---

## Comparações:

### Expected Sarsa:

```python
    def run_expected_sarsa(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1):
    """
    Algoritmo Expected-SARSA para aprendizado por reforço em ambientes com espaços de estado e ação discretos.
    """
    # Verifica se os espaços do ambiente são discretos, pois a Q-table requer indexação discreta
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    # Número de ações disponíveis no ambiente
    num_actions = env.action_space.n

    # Inicializa a Q-table com valores pequenos aleatórios para evitar empates
    Q = np.random.uniform(low=-0.01, high=0.01, size=(env.observation_space.n, num_actions))

    # Lista para armazenar a soma das recompensas por episódio
    episode_rewards = []

    # Loop principal de treinamento: itera sobre cada episódio
    for i in range(episodes):
        done = False             # Flag para controlar se o episódio terminou
        total_reward = 0         # Acumulador para o retorno total do episódio
        state, _ = env.reset()   # Reinicia o ambiente e obtém o estado inicial

        while not done:
            # Escolhe a ação usando a política epsilon-greedy
            action = epsilon_greedy_choice(Q, state, epsilon)

            # Executa a ação no ambiente e obtém o próximo estado, recompensa e flag de término
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Calcula o valor esperado do próximo estado
            if terminated:
                V_next = 0  # Se o episódio acabou, o próximo estado não tem valor
            else:
                # Obtém as probabilidades das ações no próximo estado (política epsilon-greedy)
                p_next = epsilon_greedy_probs(Q, next_state, epsilon)
                # Valor esperado do próximo estado como soma ponderada dos Q-values
                V_next = np.sum(p_next * Q[next_state])

            # Atualiza a Q-table usando a equação do Expected-SARSA
            Q[state, action] += lr * ((reward + gamma * V_next) - Q[state, action])

            # Acumula a recompensa recebida neste episódio
            total_reward += reward
            # Atualiza o estado atual para o próximo estado
            state = next_state

        # Armazena a recompensa total do episódio para análise
        episode_rewards.append(total_reward)
    
    # Retorna a lista de recompensas por episódio e a Q-table final
    return episode_rewards, Q
```


### Mt Carlo:
```python
    def run_montecarlo2(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1, render_env=None):
    """Algoritmo Monte Carlo (toda-visita) para aprendizado por reforço."""
    # Verifica se os espaços do ambiente são discretos (necessário para indexação na Q-table)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    
    # Número de ações possíveis no ambiente
    num_actions = env.action_space.n
    
    # Inicializa a Q-table com zeros
    Q = np.zeros((env.observation_space.n, num_actions))
    
    # Lista para armazenar a soma das recompensas por episódio
    episode_rewards = []
    
    # Loop principal de treinamento: percorre cada episódio
    for i in range(episodes):
        done = False               # Flag para indicar se o episódio terminou
        total_reward = 0           # Acumulador de recompensas do episódio
        trajectory = []            # Armazena a sequência de transições (estado, ação, recompensa)
        
        # Define o ambiente de treino, renderizando a cada 1000 episódios ou nos últimos 5 episódios
        train_env = render_env if (render_env is not None and ((i+1) % 1000 == 0 or i >= episodes-5)) else env
        
        # Reinicia o ambiente e obtém o estado inicial
        state, _ = train_env.reset()
        
        # -------------------------------
        # Geração da Trajetória do Episódio
        # -------------------------------
        while not done:
            # Seleciona uma ação utilizando a política epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(0, num_actions)  # Escolha aleatória
            else:
                action = np.argmax(Q[state])  # Melhor ação segundo a Q-table
            
            # Executa a ação no ambiente e obtém a transição
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated  # O episódio termina se estiver terminado ou truncado
            
            # Armazena a transição na trajetória
            trajectory.append((state, action, reward))
            
            # Acumula a recompensa total do episódio
            total_reward += reward
            
            # Atualiza o estado atual para o próximo estado
            state = next_state
        
        # Armazena a recompensa total do episódio
        episode_rewards.append(total_reward)
        
        # -------------------------------
        # Atualização Monte Carlo (toda-visita)
        # -------------------------------
        G = 0  # Inicializa o retorno descontado
        
        # Percorre a trajetória do episódio em ordem reversa (do final para o início)
        for (s, a, r) in reversed(trajectory):
            # Calcula o retorno acumulado descontado (G_t = r + gamma * G_{t+1})
            G = r + gamma * G
            
            # Atualiza a Q-table com a média incremental
            Q[s, a] += lr * (G - Q[s, a])
    
    # Retorna a lista de recompensas por episódio e a Q-table final
    return episode_rewards, Q
```

### Mescla:
```python
    def run_hybrid_expected_sarsa_mc(env, episodes, lr=0.1,   gamma=0.95, epsilon=0.1):
    # Verifica se os espaços do ambiente são discretos (necessário para indexar a Q-table)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    # Número de ações disponíveis no ambiente
    num_actions = env.action_space.n

    # Inicializa a Q-table com valores pequenos aleatórios para evitar empates
    Q = np.random.uniform(low=-0.01, high=+0.01, size=(env.observation_space.n, num_actions))

    # Lista para armazenar o retorno (soma de recompensas) de cada episódio
    all_episode_rewards = []

    # Loop principal de treinamento: itera sobre cada episódio
    for i in range(episodes):
        done = False               # Flag para controlar quando o episódio termina
        sum_rewards = 0            # Acumulador para a recompensa total do episódio
        episode_trajectory = []    # Armazena a sequência de transições (estado, ação, recompensa)

        # Reinicia o ambiente e obtém o estado inicial
        state, _ = env.reset()

        # -------------------------------
        # Parte Online: Atualização Expected-SARSA
        # -------------------------------
        while not done:
            # Seleciona uma ação utilizando a política epsilon-greedy
            action = epsilon_greedy_choice(Q, state, epsilon)

            # Executa a ação no ambiente e obtém o próximo estado, recompensa e sinal de término
            next_state, reward, terminated, truncated, _ = env.step(action)
            # O episódio termina se houver 'terminated' ou 'truncated'
            done = terminated or truncated

            # Armazena a transição para uso na atualização offline
            episode_trajectory.append((state, action, reward))

            # Se o episódio terminou, o valor do próximo estado é considerado 0
            if terminated:
                V_next_state = 0
            else:
                # Calcula as probabilidades das ações no próximo estado (epsilon-greedy)
                p_next_actions = epsilon_greedy_probs(Q, next_state, epsilon)
                # Valor esperado para o próximo estado é a soma ponderada dos Q-values
                V_next_state = np.sum(p_next_actions * Q[next_state])

            # Calcula o erro temporal (TD error) para a atualização online
            delta = (reward + gamma * V_next_state) - Q[state, action]
            # Atualiza o Q-value para a ação tomada no estado atual
            Q[state, action] += lr * delta

            # Acumula a recompensa obtida neste passo
            sum_rewards += reward
            # Atualiza o estado atual para o próximo estado
            state = next_state

        # Fim do episódio: armazena o total de recompensas obtidas
        all_episode_rewards.append(sum_rewards)

        # -------------------------------
        # Parte Offline: Atualização Monte Carlo
        # -------------------------------
        # Reinicia o retorno acumulado para o episódio (Gt)
        Gt = 0
        # Percorre a trajetória do episódio em ordem reversa (do final para o início)
        for (s, a, r) in reversed(episode_trajectory):
            # Calcula o retorno descontado (G_t = r + gamma * G_{t+1})
            Gt = r + gamma * Gt
            # Calcula a diferença entre o retorno real e a estimativa atual
            delta = Gt - Q[s, a]
            # Atualiza o Q-value usando o learning rate
            Q[s, a] += lr * delta

        # Opcional: a cada 100 episódios, imprime a média dos retornos dos últimos 100 episódios
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(all_episode_rewards[-100:])
            print(f"[Híbrido] Episódio {i+1} - Média últimos 100: {avg_reward:.3f}")

    # Retorna a lista de retornos por episódio e a Q-table final
    return all_episode_rewards, Q
```
## Resultados e Discussão

### Evolução das Recompensas

[Exemplo de Gráfico de Recompensas]
![image](https://hackmd.io/_uploads/S1b4K9cqJl.png)
![image](https://hackmd.io/_uploads/BJfaFc5qJl.png)
![image](https://hackmd.io/_uploads/B1gQ9Kq9cJg.png)


No gráfico acima (ilustrativo), observamos:

- **SARSA e TD+MC Mix:** Tendem a se estabilizar em torno de valores próximos de `-100`, indicando que o agente aprendeu a chegar ao objetivo antes de estourar o limite de passos (embora ainda não seja a rota mais curta possível).  
- **Monte Carlo:** Apresenta maior variância, com episódios em que a recompensa cai para valores como `-1200`, refletindo episódios “ruins” em que o agente explora demais ou cai repetidamente no penhasco antes de terminar o episódio.  

### Interpretação

- **Viés vs. Variância:**  
  - Monte Carlo tem menor viés (não depende de bootstrapping), mas a variância pode ser alta, especialmente nos estágios iniciais.  
  - Métodos TD, por outro lado, atualizam incrementalmente, tendendo a convergir mais rapidamente para políticas que evitam cair do penhasco.

- **Limite de Tempo :**  
  - Qualquer política que demore para chegar ao objetivo ou caia repetidamente acumula recompensas muito negativas, resultando nos “picos” de -1200

- **Mescla TD+MC:**  
  - O método híbrido se mostrou competitivo com SARSA, apresentando resultados próximos e, em alguns casos, estabilidade melhor no início do aprendizado.

---

## Observações

1. **Número de Episódios**: Em 1000 episódios, SARSA e TD+MC Mix já mostraram certa estabilidade. Entretanto, o Monte Carlo pode exigir mais episódios para convergir, dependendo da taxa de exploração e do limite de passos.
2. **Hiperparâmetros**: Fatores como `alpha (Learning Rate)`, `epsilon` e `mc_weight` influenciam diretamente a estabilidade e velocidade de aprendizado. Ainda estamos conduzindo ajustes finos.
3. **Número de Passos**: 
O parâmetro max_episode_length define o número máximo de passos que um episódio pode ter antes de ser interrompido.
Na chamada video_length=max_episode_length, ele define o comprimento máximo do vídeo gravado.
Com esse limite, se torna mais viável a comparação entre os três métodos.
4. **Optuna**:
Optuna é uma ferramenta poderosa para encontrar automaticamente os melhores hiperparâmetros.
Ele usa uma abordagem inteligente e eficiente, evitando buscas exaustivas, sendo fácil de integrar com frameworks populares de ML.

---

## Conclusões
Podemos ver que a mescla apresenta tradicionalmente um resultado médio entre os métodos apresentados, trazendo um aprendizado no equilíbrio de velocidade e estabilidade, mas possui hiperparâmetros mais sensíveis.



## Referências
https://github.com/pablo-sampaio/rl_facil/tree/main 