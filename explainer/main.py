import os
import pickle
import json
import torch
import torch.nn as nn
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from utils.ollama_utils import ensure_ollama_ready, print_ollama_status

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

class XRec:
    def __init__(self, model_name="llama3.1:8b"):
        print(f"dataset: {args.dataset}")
        print(f"model: {model_name}")

        # Check Ollama setup before initializing the model
        print("Checking Ollama setup...")
        if not ensure_ollama_ready(model_name):
            print("❌ Ollama setup failed. Please check the setup instructions.")
            print_ollama_status()
            raise RuntimeError("Ollama is not properly set up. Please run 'python setup_ollama.py' or install Ollama manually.")

        print("✅ Ollama is ready!")

        # Initialize the model with Ollama
        self.model = Explainer(model_name=model_name).to(device)
        self.data_handler = DataHandler()

        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()
        self.user_embedding_converter_path = f"./data/{args.dataset}/user_converter.pkl"
        self.item_embedding_converter_path = f"./data/{args.dataset}/item_converter.pkl"

        self.tst_predictions_path = f"./data/{args.dataset}/tst_predictions.pkl"
        self.tst_references_path = f"./data/{args.dataset}/tst_references.pkl"

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            total_loss = 0
            self.model.train()
            for i, batch in enumerate(self.trn_loader):
                user_embed, item_embed, input_text = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                input_ids, outputs, explain_pos_position = self.model.forward(user_embed, item_embed, input_text)
                input_ids = input_ids.to(device)
                explain_pos_position = explain_pos_position.to(device)
                optimizer.zero_grad()
                loss = self.model.loss(input_ids, outputs, explain_pos_position, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 100 == 0 and i != 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(self.trn_loader)}], Loss: {loss.item()}"
                    )
                    # During training, we don't generate actual text, just train the embeddings
                    print(f"Training embedding converters... (no text generation during training)")

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {total_loss}")
            # Save the model
            torch.save(
                self.model.user_embedding_converter.state_dict(),
                self.user_embedding_converter_path,
            )
            torch.save(
                self.model.item_embedding_converter.state_dict(),
                self.item_embedding_converter_path,
            )
            print(f"Saved model to {self.user_embedding_converter_path}")
            print(f"Saved model to {self.item_embedding_converter_path}")

    def evaluate(self):
        loader = self.tst_loader
        predictions_path = self.tst_predictions_path
        references_path = self.tst_references_path

        # load model
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                outputs = self.model.generate(user_embed, item_embed, input_text)
                end_idx = outputs[0].find("[")
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                predictions.append(outputs[0])
                references.append(explain[0])
                if i % 10 == 0 and i != 0:
                    print(f"Step [{i}/{len(loader)}]")
                    print(f"Generated Explanation: {outputs[0]}")

        with open(predictions_path, "wb") as file:
            pickle.dump(predictions, file)
        with open(references_path, "wb") as file:
            pickle.dump(references, file)
        print(f"Saved predictions to {predictions_path}")
        print(f"Saved references to {references_path}")   

def main():
    # Get model name from args or use default
    model_name = getattr(args, 'model_name', 'llama3.1:8b')

    try:
        sample = XRec(model_name=model_name)
        if args.mode == "finetune":
            print("Finetune model...")
            sample.train()
        elif args.mode == "generate":
            print("Generating explanations...")
            sample.evaluate()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTo fix this issue:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Start Ollama: ollama serve")
        print("3. Download model: ollama pull llama3.1:8b")
        print("4. Or run the setup script: python setup_ollama.py")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
