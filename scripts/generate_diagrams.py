"""Generate architecture diagrams for the book recommender system."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_aws_architecture_diagram():
    """Create AWS deployment architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'AWS Deployment Architecture', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Components with colors
    components = [
        # (x, y, width, height, text, color)
        (0.5, 7.5, 1.5, 0.8, 'User/Client', '#e6f2ff'),
        (2.5, 7.5, 1.5, 0.8, 'ALB\n(Load Balancer)', '#e6f2ff'),
        (4.5, 6.5, 2, 1.5, 'ECS Fargate\nAPI Container\n(Flask)', '#fff2cc'),
        (4.5, 5, 2, 0.8, 'Model Service\n(TF-IDF + CF)', '#fff2cc'),
        (7, 7, 2, 1, 'Training\n(SageMaker/Batch)\ntrain_recommender.py', '#f0e6ff'),
        (4.5, 3, 2, 1, 'S3\nRaw Data + Models\n(CSV, .pkl)', '#eaf6e0'),
        (7, 5, 2, 1, 'RDS\n(PostgreSQL)\nMetadata', '#fff0f0'),
        (1, 5, 2, 0.8, 'ElastiCache\n(Redis)\nCaching', '#f7f7e6'),
        (1, 3, 2, 0.8, 'CloudWatch\nMonitoring', '#e8ffe8'),
        (0.5, 1.5, 2, 0.8, 'CI/CD\nGitHub Actions', '#e8f0fb'),
    ]
    
    boxes = {}
    for i, (x, y, w, h, text, color) in enumerate(components):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=9, fontweight='bold')
        boxes[i] = (x + w/2, y + h/2)
    
    # Arrows (connections)
    arrows = [
        (0, 1, 'HTTP'),  # User -> ALB
        (1, 2, 'Route'),  # ALB -> ECS
        (2, 3, 'Loads'),  # ECS -> Model Service
        (3, 5, 'Read Models'),  # Model -> S3
        (6, 5, 'Write Models'),  # Training -> S3
        (2, 7, 'Query'),  # ECS -> RDS
        (2, 4, 'Cache'),  # ECS -> ElastiCache
        (9, 2, 'Deploy'),  # CI/CD -> ECS
        (8, 2, 'Metrics'),  # CloudWatch -> ECS
    ]
    
    for start, end, label in arrows:
        x1, y1 = boxes[start]
        x2, y2 = boxes[end]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               color='#2B579A', linewidth=1.5, alpha=0.7)
        ax.add_patch(arrow)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=7, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_data_flow_diagram():
    """Create data loading → transformation → training → API → frontend flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.7, 'Data Flow Pipeline: Loading → Transformation → Training → API → Frontend', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Pipeline stages
    stages = [
        # (x, y, width, height, title, details, color)
        (0.5, 3.5, 1.5, 1.5, '1. Data Loading', 
         'Kaggle Books.csv\nRatings.csv\n271K books\n1.1M ratings', '#e6f2ff'),
        (2.5, 3.5, 1.5, 1.5, '2. Transform', 
         'Clean data\nMerge tables\nFilter ratings\nNormalize', '#fff2cc'),
        (4.5, 3.5, 1.5, 1.5, '3. Training', 
         'TF-IDF matrix\nItem-Item CF\nPopularity\nSave .pkl', '#f0e6ff'),
        (6.5, 3.5, 1.5, 1.5, '4. API', 
         'Flask REST\n/recommend\nLoad models\nServe recs', '#eaf6e0'),
        (8.5, 3.5, 1.2, 1.5, '5. Frontend', 
         'UI/Mobile\nUser input\nDisplay recs', '#fff0f0'),
    ]
    
    boxes = []
    for x, y, w, h, title, details, color in stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top',
                fontsize=11, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 0.1, details, ha='center', va='center',
                fontsize=8, style='italic')
        boxes.append((x + w, y + h/2))
    
    # Flow arrows
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i]
        x2, y2 = boxes[i + 1]
        arrow = FancyArrowPatch((x1, y1), (x2 - (1.5 if i < len(boxes) - 2 else 1.2), y2),
                               arrowstyle='->', mutation_scale=25,
                               color='#2B579A', linewidth=3, alpha=0.8)
        ax.add_patch(arrow)
    
    # Add implementation details at bottom
    impl_details = [
        ('Scripts:', 'train_recommender.py, api.py, recommend_example.py', 0.5),
        ('Models:', 'tfidf_vectorizer.pkl, books_df.pkl in models/', 2.5),
        ('Algorithms:', 'Popularity, TF-IDF Content-Based, Item-Item CF', 5),
        ('Tech Stack:', 'Python, pandas, scikit-learn, Flask, scipy', 7.5),
    ]
    
    for label, text, x in impl_details:
        ax.text(x, 2.5, f'{label}', ha='left', va='top',
                fontsize=8, fontweight='bold')
        ax.text(x, 2.2, text, ha='left', va='top',
                fontsize=7, style='italic', wrap=True)
    
    # Feedback loop - from Frontend (right) back to Data Loading (left) below the pipeline
    feedback_arrow = FancyArrowPatch((8.5, 3.4), (2, 3.4),
                                    arrowstyle='->', mutation_scale=25,
                                    color='#A00000', linewidth=2.5, 
                                    linestyle='dashed', alpha=0.7,
                                    connectionstyle="arc3,rad=-0.5")
    ax.add_patch(feedback_arrow)
    ax.text(5.2, 2.8, 'User Feedback Loop → Retrain (Future)', 
            ha='center', va='center', fontsize=10, color='#A00000', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#A00000', linestyle='dashed', linewidth=1.5))
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save both diagrams."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate AWS Architecture
    print("Generating AWS architecture diagram...")
    fig1 = create_aws_architecture_diagram()
    aws_path = os.path.join(output_dir, 'architecture_aws.png')
    fig1.savefig(aws_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {aws_path}")
    plt.close(fig1)
    
    # Generate Data Flow
    print("Generating data flow diagram...")
    fig2 = create_data_flow_diagram()
    flow_path = os.path.join(output_dir, 'architecture_dataflow.png')
    fig2.savefig(flow_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {flow_path}")
    plt.close(fig2)
    
    print("\n✓ Both diagrams generated successfully!")
    print(f"  - {aws_path}")
    print(f"  - {flow_path}")

if __name__ == '__main__':
    main()
