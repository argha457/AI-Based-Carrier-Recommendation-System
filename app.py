import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ModernCourseRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Course Recommendation System")
        self.root.geometry("1000x800")
        
        # Set theme colors
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#64B5F6',
            'background': '#F5F5F5',
            'text': '#212121',
            'accent': '#1976D2'
        }
        
        # Configure styles
        self.configure_styles()
        
        # Define all features
        self.features = ['Drawing', 'Dancing', 'Singing', 'Sports', 'Video Game', 'Acting', 'Travelling', 
                        'Gardening', 'Animals', 'Photography', 'Teaching', 'Exercise', 'Coding', 
                        'Electricity Components', 'Mechanic Parts', 'Computer Parts', 'Researching', 
                        'Architecture', 'Historic Collection', 'Botany', 'Zoology', 'Physics', 'Accounting', 
                        'Economics', 'Sociology', 'Geography', 'Psycology', 'History', 'Science', 
                        'Bussiness Education', 'Chemistry', 'Mathematics', 'Biology', 'Makeup', 'Designing', 
                        'Content writing', 'Crafting', 'Literature', 'Reading', 'Cartooning', 'Debating', 
                        'Asrtology', 'Hindi', 'French', 'English', 'Urdu', 'Other Language', 'Solving Puzzles', 
                        'Gymnastics', 'Yoga', 'Engeeniering', 'Doctor', 'Pharmisist', 'Cycling', 'Knitting', 
                        'Director', 'Journalism', 'Bussiness', 'Listening Music']
        
        # Updated feature categories to include all features
        self.feature_categories = {
            'Arts & Creativity': ['Drawing', 'Dancing', 'Singing', 'Acting', 'Photography', 'Makeup', 'Designing', 
                                'Crafting', 'Cartooning', 'Knitting', 'Listening Music'],
            'Technology & Engineering': ['Video Game', 'Coding', 'Computer Parts', 'Electricity Components', 
                                      'Mechanic Parts', 'Engeeniering'],
            'Sciences': ['Physics', 'Chemistry', 'Biology', 'Mathematics', 'Botany', 'Zoology', 'Science'],
            'Medicine & Health': ['Doctor', 'Pharmisist'],
            'Humanities': ['History', 'Literature', 'Geography', 'Sociology', 'Psycology', 'Historic Collection'],
            'Languages': ['Hindi', 'French', 'English', 'Urdu', 'Other Language'],
            'Business & Economics': ['Accounting', 'Economics', 'Bussiness Education', 'Bussiness'],
            'Physical Activities': ['Sports', 'Exercise', 'Gymnastics', 'Yoga', 'Cycling'],
            'Professional Skills': ['Teaching', 'Researching', 'Architecture', 'Content writing', 'Director', 
                                  'Journalism', 'Debating'],
            'Hobbies & Interests': ['Travelling', 'Gardening', 'Animals', 'Reading', 'Asrtology', 
                                  'Solving Puzzles']
        }

        # Try to load the model
        self.model_loaded = False
        try:
            self.model = joblib.load('model .pkl')
            self.model_loaded = True
        except:
            try:
                self.load_and_train_initial()
            except:
                messagebox.showwarning("Model Not Found", 
                    "No model found. Please go to File → Load Data & Train Model to load your data file.")

        self.create_widgets()

    def configure_styles(self):
        """Configure ttk styles for widgets"""
        style = ttk.Style()
        
        # Configure main styles
        style.configure('Header.TLabel', 
                       font=('Helvetica', 24, 'bold'),
                       foreground=self.colors['primary'])
        
        style.configure('SubHeader.TLabel',
                       font=('Helvetica', 12),
                       foreground=self.colors['text'])
        
        style.configure('Category.TLabelframe',
                       background=self.colors['background'])
        
        style.configure('Category.TLabelframe.Label',
                       font=('Helvetica', 11, 'bold'),
                       foreground=self.colors['primary'])
        
        style.configure('Action.TButton',
                       font=('Helvetica', 11),
                       padding=10)
        
        style.configure('Result.TFrame',
                       background=self.colors['background'])
        
        style.configure('Modern.TCheckbutton',
                       font=('Helvetica', 10),
                       background=self.colors['background'])

    def create_widgets(self):
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="20")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create menu bar
        self.create_menu()

        # Create header
        header = ttk.Label(self.main_container, 
                          text="Course Recommendation System",
                          style='Header.TLabel')
        header.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create description
        description = ttk.Label(self.main_container,
                              text="Select your interests in each category to get personalized course recommendations",
                              style='SubHeader.TLabel')
        description.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        # Create notebook for categorized features
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))

        # Create checkboxes dictionary
        self.checkboxes = {}

        # Initialize checkboxes for ALL features first
        for feature in self.features:
            self.checkboxes[feature] = tk.BooleanVar()

        # Create tabs for each category
        for category, features in self.feature_categories.items():
            tab = ttk.Frame(self.notebook, padding="20")
            self.notebook.add(tab, text=category)
            
            # Create grid of checkboxes
            for i, feature in enumerate(features):
                row = i // 3
                col = i % 3
                
                checkbox = ttk.Checkbutton(tab, 
                                         text=feature,
                                         variable=self.checkboxes[feature],
                                         style='Modern.TCheckbutton')
                checkbox.grid(row=row, column=col, padx=20, pady=5, sticky=tk.W)

        # Create button container
        button_container = ttk.Frame(self.main_container)
        button_container.grid(row=3, column=0, columnspan=2, pady=(0, 20))

        # Create action buttons
        select_all_btn = ttk.Button(button_container,
                                  text="Select All",
                                  style='Action.TButton',
                                  command=self.select_all)
        select_all_btn.pack(side=tk.LEFT, padx=5)

        clear_all_btn = ttk.Button(button_container,
                                 text="Clear All",
                                 style='Action.TButton',
                                 command=self.clear_all)
        clear_all_btn.pack(side=tk.LEFT, padx=5)

        recommend_btn = ttk.Button(button_container,
                                 text="Get Recommendation",
                                 style='Action.TButton',
                                 command=self.make_prediction)
        recommend_btn.pack(side=tk.LEFT, padx=5)

        # Create result frame
        self.result_frame = ttk.Frame(self.main_container, style='Result.TFrame')
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.result_label = ttk.Label(self.result_frame,
                                    text="Your recommended course will appear here",
                                    font=('Helvetica', 14),
                                    wraplength=800,
                                    justify=tk.CENTER)
        self.result_label.pack(pady=20)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data & Train Model", command=self.load_and_train)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def select_all(self):
        """Select all checkboxes"""
        for var in self.checkboxes.values():
            var.set(True)

    def clear_all(self):
        """Clear all checkboxes"""
        for var in self.checkboxes.values():
            var.set(False)

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
            "Course Recommendation System\n\n"
            "This system uses machine learning to recommend suitable courses "
            "based on your interests and preferences.\n\n"
            "Version 2.0")

    def load_and_train_initial(self):
        """Attempt to train model with stud.csv in current directory"""
        if os.path.exists('stud.csv'):
            data = pd.read_csv('stud.csv')
            
            label_encoder = LabelEncoder()
            data['Courses_label'] = label_encoder.fit_transform(data['Courses'])
            
            for col in self.features:
                if col in data.columns:
                    data[col] = label_encoder.fit_transform(data[col])
            
            X = data[self.features]
            y = data['Courses_label']
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            joblib.dump(self.model, 'model .pkl')
            self.model_loaded = True
            messagebox.showinfo("Success", "Model trained automatically with stud.csv!")

    def load_and_train(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if not file_path:
                return

            data = pd.read_csv(file_path)
            
            label_encoder = LabelEncoder()
            data['Courses_label'] = label_encoder.fit_transform(data['Courses'])
            
            for col in self.features:
                if col in data.columns:
                    data[col] = label_encoder.fit_transform(data[col])
            
            X = data[self.features]
            y = data['Courses_label']
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            joblib.dump(self.model, 'model .pkl')
            self.model_loaded = True
            
            messagebox.showinfo("Success", "Model trained and saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def make_prediction(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load data and train the model first!")
            return

        user_input = {}
        for feature in self.features:
            user_input[feature] = 1 if self.checkboxes[feature].get() else 0

        user_data = pd.DataFrame([user_input])

        prediction = self.model.predict(user_data)
        
        numeric_to_category = {
            0: 'Animation, Graphics and Multimedia',
            1: 'B.Arch- Bachelor of Architecture',
            2: 'B.Com- Bachelor of Commerce',
            3: 'B.Ed.',
            4: 'B.Sc- Applied Geology',
            5: 'B.Sc- Nursing',
            6: 'B.Sc. Chemistry',
            7: 'B.Sc. Mathematics',
            8: 'B.Sc.- Information Technology',
            9: 'B.Sc.- Physics',
            10: 'B.Tech.-Civil Engineering',
            11: 'B.Tech.-Computer Science and Engineering',
            12: 'B.Tech.-Electrical and Electronics Engineering',
            13: 'B.Tech.-Electronics and Communication Engineering',
            14: 'B.Tech.-Mechanical Engineering',
            15: 'BA in Economics',
            16: 'BA in English',
            17: 'BA in Hindi',
            18: 'BA in History',
            19: 'BBA- Bachelor of Business Administration',
            20: 'BBS- Bachelor of Business Studies',
            21: 'BCA- Bachelor of Computer Applications',
            22: 'BDS- Bachelor of Dental Surgery',
            23: 'BEM- Bachelor of Event Management',
            24: 'BFD- Bachelor of Fashion Designing',
            25: 'BJMC- Bachelor of Journalism and Mass Communication',
            26: 'BPharma- Bachelor of Pharmacy',
            27: 'BTTM- Bachelor of Travel and Tourism Management',
            28: 'BVA- Bachelor of Visual Arts',
            29: 'CA- Chartered Accountancy',
            30: 'CS- Company Secretary',
            31: 'Civil Services',
            32: 'Diploma in Dramatic Arts',
            33: 'Integrated Law Course- BA + LL.B',
            34: 'MBBS'
        }

        numeric_prediction = prediction[0]
        categorical_prediction = numeric_to_category.get(numeric_prediction, "Unknown")

        self.result_label.config(
            text=f"Based on your interests, we recommend:\n\n{categorical_prediction}",
            foreground=self.colors['primary']
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernCourseRecommenderGUI(root)
    root.mainloop()