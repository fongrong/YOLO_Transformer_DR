"""
Stock Watcher Application for Windows
A simple GUI application to monitor stock prices in real-time.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import json
import os
from datetime import datetime
from typing import Dict, Optional, Callable
import urllib.request
import urllib.error


class StockDataFetcher:
    """Fetches stock data from Yahoo Finance API."""

    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current stock price for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

        Returns:
            Dictionary with stock data or None if fetch fails
        """
        try:
            url = f"{self.base_url}{symbol}?interval=1m&range=1d"
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())

            result = data.get('chart', {}).get('result', [])
            if not result:
                return None

            meta = result[0].get('meta', {})
            indicators = result[0].get('indicators', {}).get('quote', [{}])[0]

            current_price = meta.get('regularMarketPrice', 0)
            previous_close = meta.get('previousClose', 0)

            change = current_price - previous_close if previous_close else 0
            change_percent = (change / previous_close * 100) if previous_close else 0

            return {
                'symbol': symbol.upper(),
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'high': meta.get('regularMarketDayHigh', 0),
                'low': meta.get('regularMarketDayLow', 0),
                'volume': meta.get('regularMarketVolume', 0),
                'market_state': meta.get('marketState', 'UNKNOWN'),
                'currency': meta.get('currency', 'USD'),
                'exchange': meta.get('exchangeName', 'N/A'),
            }

        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None


class Alert:
    """Represents a price alert for a stock."""

    def __init__(self, symbol: str, target_price: float, alert_type: str, callback: Callable):
        self.symbol = symbol.upper()
        self.target_price = target_price
        self.alert_type = alert_type  # 'above' or 'below'
        self.callback = callback
        self.triggered = False

    def check(self, current_price: float) -> bool:
        """Check if alert condition is met."""
        if self.triggered:
            return False

        if self.alert_type == 'above' and current_price >= self.target_price:
            self.triggered = True
            self.callback(self)
            return True
        elif self.alert_type == 'below' and current_price <= self.target_price:
            self.triggered = True
            self.callback(self)
            return True
        return False


class StockWatcherApp:
    """Main Stock Watcher Application."""

    CONFIG_FILE = "stock_watcher_config.json"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Watcher")
        self.root.geometry("900x600")
        self.root.minsize(700, 400)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Data
        self.watched_stocks: Dict[str, Dict] = {}
        self.alerts: list = []
        self.fetcher = StockDataFetcher()
        self.update_interval = 30000  # 30 seconds
        self.is_running = True

        # Build UI
        self._create_menu()
        self._create_toolbar()
        self._create_main_view()
        self._create_status_bar()

        # Load saved configuration
        self._load_config()

        # Start auto-refresh
        self._schedule_update()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Stock", command=self._add_stock_dialog)
        file_menu.add_command(label="Remove Stock", command=self._remove_stock)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        # Alerts menu
        alerts_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Alerts", menu=alerts_menu)
        alerts_menu.add_command(label="Add Alert", command=self._add_alert_dialog)
        alerts_menu.add_command(label="View Alerts", command=self._view_alerts)
        alerts_menu.add_command(label="Clear Alerts", command=self._clear_alerts)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Refresh Interval", command=self._set_refresh_interval)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_toolbar(self):
        """Create toolbar with buttons."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Add stock button
        add_btn = ttk.Button(toolbar, text="+ Add Stock", command=self._add_stock_dialog)
        add_btn.pack(side=tk.LEFT, padx=2)

        # Remove stock button
        remove_btn = ttk.Button(toolbar, text="- Remove", command=self._remove_stock)
        remove_btn.pack(side=tk.LEFT, padx=2)

        # Refresh button
        refresh_btn = ttk.Button(toolbar, text="↻ Refresh", command=self._refresh_all)
        refresh_btn.pack(side=tk.LEFT, padx=2)

        # Add alert button
        alert_btn = ttk.Button(toolbar, text="⚠ Add Alert", command=self._add_alert_dialog)
        alert_btn.pack(side=tk.LEFT, padx=2)

        # Search entry
        ttk.Label(toolbar, text="Quick Add:").pack(side=tk.LEFT, padx=(20, 5))
        self.quick_add_entry = ttk.Entry(toolbar, width=15)
        self.quick_add_entry.pack(side=tk.LEFT, padx=2)
        self.quick_add_entry.bind('<Return>', lambda e: self._quick_add_stock())

        quick_add_btn = ttk.Button(toolbar, text="Add", command=self._quick_add_stock)
        quick_add_btn.pack(side=tk.LEFT, padx=2)

    def _create_main_view(self):
        """Create main stock list view."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview for stock list
        columns = ('symbol', 'price', 'change', 'change_pct', 'high', 'low', 'volume', 'market')
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings', selectmode='browse')

        # Define headings
        self.tree.heading('symbol', text='Symbol', anchor=tk.W)
        self.tree.heading('price', text='Price', anchor=tk.E)
        self.tree.heading('change', text='Change', anchor=tk.E)
        self.tree.heading('change_pct', text='Change %', anchor=tk.E)
        self.tree.heading('high', text='High', anchor=tk.E)
        self.tree.heading('low', text='Low', anchor=tk.E)
        self.tree.heading('volume', text='Volume', anchor=tk.E)
        self.tree.heading('market', text='Market', anchor=tk.CENTER)

        # Define column widths
        self.tree.column('symbol', width=80, minwidth=60)
        self.tree.column('price', width=100, minwidth=80)
        self.tree.column('change', width=100, minwidth=80)
        self.tree.column('change_pct', width=100, minwidth=80)
        self.tree.column('high', width=100, minwidth=80)
        self.tree.column('low', width=100, minwidth=80)
        self.tree.column('volume', width=120, minwidth=100)
        self.tree.column('market', width=80, minwidth=60)

        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Pack
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure tags for coloring
        self.tree.tag_configure('positive', foreground='green')
        self.tree.tag_configure('negative', foreground='red')
        self.tree.tag_configure('neutral', foreground='gray')

    def _create_status_bar(self):
        """Create status bar at the bottom."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.last_update_label = ttk.Label(self.status_frame, text="", anchor=tk.E)
        self.last_update_label.pack(side=tk.RIGHT, padx=5)

    def _add_stock_dialog(self):
        """Show dialog to add a new stock."""
        symbol = simpledialog.askstring(
            "Add Stock",
            "Enter stock symbol (e.g., AAPL, GOOGL, MSFT):",
            parent=self.root
        )
        if symbol:
            self._add_stock(symbol.strip().upper())

    def _quick_add_stock(self):
        """Add stock from quick add entry."""
        symbol = self.quick_add_entry.get().strip().upper()
        if symbol:
            self._add_stock(symbol)
            self.quick_add_entry.delete(0, tk.END)

    def _add_stock(self, symbol: str):
        """Add a stock to the watch list."""
        if not symbol:
            return

        if symbol in self.watched_stocks:
            messagebox.showinfo("Info", f"{symbol} is already in your watch list.")
            return

        self._set_status(f"Fetching data for {symbol}...")

        # Fetch in background thread
        def fetch_and_add():
            data = self.fetcher.get_stock_price(symbol)
            self.root.after(0, lambda: self._on_stock_fetched(symbol, data))

        threading.Thread(target=fetch_and_add, daemon=True).start()

    def _on_stock_fetched(self, symbol: str, data: Optional[Dict]):
        """Handle fetched stock data."""
        if data:
            self.watched_stocks[symbol] = data
            self._update_tree_item(symbol, data)
            self._set_status(f"Added {symbol} to watch list")
            self._save_config()
        else:
            messagebox.showerror("Error", f"Could not fetch data for {symbol}. Please check the symbol.")
            self._set_status("Ready")

    def _remove_stock(self):
        """Remove selected stock from watch list."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a stock to remove.")
            return

        item = self.tree.item(selection[0])
        symbol = item['values'][0]

        if messagebox.askyesno("Confirm", f"Remove {symbol} from watch list?"):
            self.tree.delete(selection[0])
            del self.watched_stocks[symbol]
            self._save_config()
            self._set_status(f"Removed {symbol}")

    def _update_tree_item(self, symbol: str, data: Dict):
        """Update or insert tree item with stock data."""
        # Format values
        price = f"${data['price']:.2f}"
        change = f"${data['change']:+.2f}"
        change_pct = f"{data['change_percent']:+.2f}%"
        high = f"${data['high']:.2f}"
        low = f"${data['low']:.2f}"
        volume = f"{data['volume']:,}"
        market = data['market_state']

        values = (symbol, price, change, change_pct, high, low, volume, market)

        # Determine tag based on change
        if data['change'] > 0:
            tag = 'positive'
        elif data['change'] < 0:
            tag = 'negative'
        else:
            tag = 'neutral'

        # Check if item exists
        existing_items = self.tree.get_children()
        for item_id in existing_items:
            if self.tree.item(item_id)['values'][0] == symbol:
                self.tree.item(item_id, values=values, tags=(tag,))
                return

        # Insert new item
        self.tree.insert('', tk.END, values=values, tags=(tag,))

    def _refresh_all(self):
        """Refresh all stock data."""
        if not self.watched_stocks:
            self._set_status("No stocks to refresh")
            return

        self._set_status("Refreshing...")

        def refresh_thread():
            for symbol in list(self.watched_stocks.keys()):
                if not self.is_running:
                    break
                data = self.fetcher.get_stock_price(symbol)
                if data:
                    self.watched_stocks[symbol] = data
                    self.root.after(0, lambda s=symbol, d=data: self._update_tree_item(s, d))
                    # Check alerts
                    self._check_alerts(symbol, data['price'])

            self.root.after(0, self._on_refresh_complete)

        threading.Thread(target=refresh_thread, daemon=True).start()

    def _on_refresh_complete(self):
        """Called when refresh is complete."""
        now = datetime.now().strftime("%H:%M:%S")
        self._set_status("Ready")
        self.last_update_label.config(text=f"Last update: {now}")

    def _schedule_update(self):
        """Schedule next automatic update."""
        if self.is_running:
            self._refresh_all()
            self.root.after(self.update_interval, self._schedule_update)

    def _add_alert_dialog(self):
        """Show dialog to add a price alert."""
        if not self.watched_stocks:
            messagebox.showinfo("Info", "Please add some stocks first.")
            return

        # Create alert dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Price Alert")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        # Symbol selection
        ttk.Label(dialog, text="Stock:").pack(pady=(10, 0))
        symbol_var = tk.StringVar()
        symbol_combo = ttk.Combobox(dialog, textvariable=symbol_var, values=list(self.watched_stocks.keys()))
        symbol_combo.pack(pady=5)
        if self.watched_stocks:
            symbol_combo.current(0)

        # Alert type
        ttk.Label(dialog, text="Alert when price goes:").pack(pady=(10, 0))
        alert_type_var = tk.StringVar(value='above')
        ttk.Radiobutton(dialog, text="Above", variable=alert_type_var, value='above').pack()
        ttk.Radiobutton(dialog, text="Below", variable=alert_type_var, value='below').pack()

        # Target price
        ttk.Label(dialog, text="Target Price ($):").pack(pady=(10, 0))
        price_entry = ttk.Entry(dialog)
        price_entry.pack(pady=5)

        def add_alert():
            try:
                symbol = symbol_var.get()
                target_price = float(price_entry.get())
                alert_type = alert_type_var.get()

                alert = Alert(symbol, target_price, alert_type, self._on_alert_triggered)
                self.alerts.append(alert)

                messagebox.showinfo("Success", f"Alert added: {symbol} {alert_type} ${target_price:.2f}")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid price.")

        ttk.Button(dialog, text="Add Alert", command=add_alert).pack(pady=10)

    def _on_alert_triggered(self, alert: Alert):
        """Handle triggered alert."""
        self.root.after(0, lambda: messagebox.showwarning(
            "Price Alert",
            f"🔔 {alert.symbol} has gone {alert.alert_type} ${alert.target_price:.2f}!\n"
            f"Current price: ${self.watched_stocks.get(alert.symbol, {}).get('price', 0):.2f}"
        ))

    def _check_alerts(self, symbol: str, current_price: float):
        """Check all alerts for a symbol."""
        for alert in self.alerts:
            if alert.symbol == symbol and not alert.triggered:
                alert.check(current_price)

    def _view_alerts(self):
        """Show current alerts."""
        if not self.alerts:
            messagebox.showinfo("Alerts", "No alerts set.")
            return

        alert_text = "Current Alerts:\n\n"
        for i, alert in enumerate(self.alerts, 1):
            status = "✓ Triggered" if alert.triggered else "Active"
            alert_text += f"{i}. {alert.symbol} {alert.alert_type} ${alert.target_price:.2f} [{status}]\n"

        messagebox.showinfo("Alerts", alert_text)

    def _clear_alerts(self):
        """Clear all alerts."""
        if self.alerts and messagebox.askyesno("Confirm", "Clear all alerts?"):
            self.alerts.clear()
            messagebox.showinfo("Info", "All alerts cleared.")

    def _set_refresh_interval(self):
        """Set the auto-refresh interval."""
        current_seconds = self.update_interval // 1000
        new_interval = simpledialog.askinteger(
            "Refresh Interval",
            "Enter refresh interval in seconds (10-300):",
            parent=self.root,
            initialvalue=current_seconds,
            minvalue=10,
            maxvalue=300
        )
        if new_interval:
            self.update_interval = new_interval * 1000
            messagebox.showinfo("Info", f"Refresh interval set to {new_interval} seconds.")

    def _set_status(self, message: str):
        """Update status bar message."""
        self.status_label.config(text=message)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Stock Watcher",
            "Stock Watcher v1.0\n\n"
            "A simple application to monitor stock prices.\n\n"
            "Features:\n"
            "• Real-time stock price monitoring\n"
            "• Price change alerts\n"
            "• Auto-refresh functionality\n\n"
            "Data provided by Yahoo Finance."
        )

    def _save_config(self):
        """Save configuration to file."""
        config = {
            'watched_symbols': list(self.watched_stocks.keys()),
            'update_interval': self.update_interval
        }
        try:
            config_path = os.path.join(os.path.dirname(__file__), self.CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _load_config(self):
        """Load configuration from file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), self.CONFIG_FILE)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

                self.update_interval = config.get('update_interval', 30000)

                # Load saved stocks
                for symbol in config.get('watched_symbols', []):
                    self._add_stock(symbol)
        except Exception as e:
            print(f"Error loading config: {e}")

    def _on_close(self):
        """Handle application close."""
        self.is_running = False
        self._save_config()
        self.root.destroy()

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = StockWatcherApp()
    app.run()


if __name__ == "__main__":
    main()
