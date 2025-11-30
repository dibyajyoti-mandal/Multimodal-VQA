import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax[0].plot(epochs, history['eval_loss'], label='Eval Loss', marker='s')
    ax[0].set_title('Training and Evaluation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, history['bleu_score'], label='BLEU Score', marker='d', color='g')
    ax[1].set_title('BLEU Score Over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('BLEU Score')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
