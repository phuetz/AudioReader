"""
Normalisation avancee du texte pour TTS haute qualite.

Fonctionnalites:
- Conversion des nombres en mots
- Expansion des dates et heures
- Gestion des abreviations
- Normalisation des unites
- Support des chiffres romains
- Gestion des symboles et emojis
"""
import re
from typing import Optional


class NumberToWords:
    """Convertit les nombres en mots (francais et anglais)."""

    # Francais
    UNITS_FR = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
    TENS_FR = ["", "dix", "vingt", "trente", "quarante", "cinquante", "soixante", "soixante", "quatre-vingt", "quatre-vingt"]
    TEENS_FR = ["dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"]

    # Anglais
    UNITS_EN = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    TENS_EN = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    TEENS_EN = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]

    def __init__(self, lang: str = "fr"):
        self.lang = lang
        if lang == "fr":
            self.units = self.UNITS_FR
            self.tens = self.TENS_FR
            self.teens = self.TEENS_FR
        else:
            self.units = self.UNITS_EN
            self.tens = self.TENS_EN
            self.teens = self.TEENS_EN

    def convert(self, n: int) -> str:
        """Convertit un nombre en mots."""
        if n == 0:
            return "zero" if self.lang == "en" else "zero"
        if n < 0:
            prefix = "minus " if self.lang == "en" else "moins "
            return prefix + self.convert(-n)

        if n < 10:
            return self.units[n]
        elif n < 20:
            return self.teens[n - 10]
        elif n < 100:
            return self._convert_tens(n)
        elif n < 1000:
            return self._convert_hundreds(n)
        elif n < 1000000:
            return self._convert_thousands(n)
        elif n < 1000000000:
            return self._convert_millions(n)
        else:
            return self._convert_billions(n)

    def _convert_tens(self, n: int) -> str:
        """Convertit un nombre entre 20 et 99."""
        if self.lang == "fr":
            # Cas speciaux francais (70, 80, 90)
            if 70 <= n < 80:
                return "soixante-" + self.teens[n - 70]
            elif n == 80:
                return "quatre-vingts"
            elif 80 < n < 100:
                return "quatre-vingt-" + (self.teens[n - 90] if n >= 90 else self.units[n - 80])
            else:
                tens_digit = n // 10
                unit_digit = n % 10
                if unit_digit == 0:
                    return self.tens[tens_digit]
                elif unit_digit == 1 and tens_digit in [2, 3, 4, 5, 6]:
                    return self.tens[tens_digit] + "-et-un"
                else:
                    return self.tens[tens_digit] + "-" + self.units[unit_digit]
        else:
            tens_digit = n // 10
            unit_digit = n % 10
            if unit_digit == 0:
                return self.tens[tens_digit]
            return self.tens[tens_digit] + "-" + self.units[unit_digit]

    def _convert_hundreds(self, n: int) -> str:
        """Convertit un nombre entre 100 et 999."""
        hundreds = n // 100
        remainder = n % 100

        if self.lang == "fr":
            if hundreds == 1:
                prefix = "cent"
            else:
                prefix = self.units[hundreds] + " cent"
                if remainder == 0:
                    prefix += "s"
        else:
            prefix = self.units[hundreds] + " hundred"

        if remainder == 0:
            return prefix
        return prefix + " " + self.convert(remainder)

    def _convert_thousands(self, n: int) -> str:
        """Convertit un nombre entre 1000 et 999999."""
        thousands = n // 1000
        remainder = n % 1000

        if self.lang == "fr":
            if thousands == 1:
                prefix = "mille"
            else:
                prefix = self.convert(thousands) + " mille"
        else:
            prefix = self.convert(thousands) + " thousand"

        if remainder == 0:
            return prefix
        return prefix + " " + self.convert(remainder)

    def _convert_millions(self, n: int) -> str:
        """Convertit un nombre entre 1M et 999M."""
        millions = n // 1000000
        remainder = n % 1000000

        if self.lang == "fr":
            word = "million" if millions == 1 else "millions"
        else:
            word = "million"

        prefix = self.convert(millions) + " " + word

        if remainder == 0:
            return prefix
        return prefix + " " + self.convert(remainder)

    def _convert_billions(self, n: int) -> str:
        """Convertit un nombre >= 1 milliard."""
        billions = n // 1000000000
        remainder = n % 1000000000

        if self.lang == "fr":
            word = "milliard" if billions == 1 else "milliards"
        else:
            word = "billion"

        prefix = self.convert(billions) + " " + word

        if remainder == 0:
            return prefix
        return prefix + " " + self.convert(remainder)


class TextNormalizer:
    """
    Normalise le texte pour une synthese vocale optimale.

    Transforme les elements non-verbaux en texte prononable.
    """

    # Mois en francais et anglais
    MONTHS_FR = {
        "01": "janvier", "02": "fevrier", "03": "mars", "04": "avril",
        "05": "mai", "06": "juin", "07": "juillet", "08": "aout",
        "09": "septembre", "10": "octobre", "11": "novembre", "12": "decembre",
        "1": "janvier", "2": "fevrier", "3": "mars", "4": "avril",
        "5": "mai", "6": "juin", "7": "juillet", "8": "aout",
        "9": "septembre"
    }

    MONTHS_EN = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December",
        "1": "January", "2": "February", "3": "March", "4": "April",
        "5": "May", "6": "June", "7": "July", "8": "August",
        "9": "September"
    }

    # Chiffres romains
    ROMAN_VALUES = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }

    # Abreviations courantes
    ABBREVIATIONS_FR = {
        "M.": "Monsieur",
        "Mme": "Madame",
        "Mlle": "Mademoiselle",
        "Dr": "Docteur",
        "Dr.": "Docteur",
        "Pr": "Professeur",
        "Pr.": "Professeur",
        "Me": "Maitre",
        "St": "Saint",
        "Ste": "Sainte",
        "etc.": "et cetera",
        "cf.": "confer",
        "ex.": "exemple",
        "env.": "environ",
        "min.": "minutes",
        "max.": "maximum",
        "nb": "nombre",
        "n°": "numero",
        "No": "numero",
        "p.": "page",
        "pp.": "pages",
        "vol.": "volume",
        "chap.": "chapitre",
        "fig.": "figure",
        "tel.": "telephone",
        "fax": "fax",
    }

    ABBREVIATIONS_EN = {
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miz",
        "Dr.": "Doctor",
        "Prof.": "Professor",
        "St.": "Saint",
        "etc.": "et cetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "approx.": "approximately",
        "govt.": "government",
        "dept.": "department",
        "min.": "minutes",
        "max.": "maximum",
        "no.": "number",
        "p.": "page",
        "pp.": "pages",
        "vol.": "volume",
        "ch.": "chapter",
        "fig.": "figure",
        "tel.": "telephone",
    }

    # Symboles
    SYMBOLS = {
        "&": " et ",
        "@": " arobase ",
        "#": " hashtag ",
        "~": " environ ",
        "+": " plus ",
        "=": " egale ",
        "<": " inferieur a ",
        ">": " superieur a ",
        "°": " degres ",
        "©": " copyright ",
        "®": " marque deposee ",
        "™": " trademark ",
        "§": " paragraphe ",
        "¶": " paragraphe ",
        "†": "",
        "‡": "",
        "•": "",
        "→": " fleche ",
        "←": " fleche ",
        "↑": " fleche ",
        "↓": " fleche ",
    }

    SYMBOLS_EN = {
        "&": " and ",
        "@": " at ",
        "#": " hashtag ",
        "~": " approximately ",
        "+": " plus ",
        "=": " equals ",
        "<": " less than ",
        ">": " greater than ",
        "°": " degrees ",
        "©": " copyright ",
        "®": " registered ",
        "™": " trademark ",
        "§": " section ",
        "¶": " paragraph ",
    }

    # Unites
    UNITS_FR = {
        "km": "kilometres",
        "m": "metres",
        "cm": "centimetres",
        "mm": "millimetres",
        "kg": "kilogrammes",
        "g": "grammes",
        "mg": "milligrammes",
        "l": "litres",
        "ml": "millilitres",
        "h": "heures",
        "min": "minutes",
        "s": "secondes",
        "ms": "millisecondes",
        "km/h": "kilometres heure",
        "m/s": "metres par seconde",
    }

    UNITS_EN = {
        "km": "kilometers",
        "m": "meters",
        "cm": "centimeters",
        "mm": "millimeters",
        "kg": "kilograms",
        "g": "grams",
        "mg": "milligrams",
        "l": "liters",
        "ml": "milliliters",
        "h": "hours",
        "min": "minutes",
        "s": "seconds",
        "ms": "milliseconds",
        "mph": "miles per hour",
        "ft": "feet",
        "in": "inches",
        "lb": "pounds",
        "oz": "ounces",
    }

    def __init__(self, lang: str = "fr"):
        self.lang = lang
        self.number_converter = NumberToWords(lang)
        self.months = self.MONTHS_FR if lang == "fr" else self.MONTHS_EN
        self.abbreviations = self.ABBREVIATIONS_FR if lang == "fr" else self.ABBREVIATIONS_EN
        self.symbols = self.SYMBOLS if lang == "fr" else self.SYMBOLS_EN
        self.units = self.UNITS_FR if lang == "fr" else self.UNITS_EN

    def normalize(self, text: str) -> str:
        """Applique toutes les normalisations."""
        # Ordre important!
        text = self._normalize_urls(text)
        text = self._normalize_emails(text)
        text = self._normalize_dates(text)
        text = self._normalize_times(text)
        text = self._normalize_phone_numbers(text)
        text = self._normalize_roman_numerals(text)
        text = self._normalize_numbers_with_units(text)
        text = self._normalize_percentages(text)
        text = self._normalize_currency(text)
        text = self._normalize_ordinals(text)
        text = self._normalize_numbers(text)
        text = self._normalize_abbreviations(text)
        text = self._normalize_symbols(text)
        text = self._normalize_whitespace(text)
        return text

    def _normalize_urls(self, text: str) -> str:
        """Remplace les URLs par une description."""
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        if self.lang == "fr":
            return re.sub(pattern, " lien internet ", text)
        return re.sub(pattern, " web link ", text)

    def _normalize_emails(self, text: str) -> str:
        """Normalise les adresses email."""
        def replace_email(match):
            email = match.group(0)
            parts = email.split('@')
            if len(parts) == 2:
                user, domain = parts
                arobase = " arobase " if self.lang == "fr" else " at "
                return user + arobase + domain.replace('.', ' point ')
            return email

        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.sub(pattern, replace_email, text)

    def _normalize_dates(self, text: str) -> str:
        """Convertit les dates en texte."""
        # Format JJ/MM/AAAA ou JJ-MM-AAAA
        def replace_date(match):
            day = match.group(1).lstrip('0')
            month = match.group(2)
            year = match.group(3)

            month_name = self.months.get(month, month)
            day_word = self.number_converter.convert(int(day))

            if self.lang == "fr":
                if day == "1":
                    day_word = "premier"
                return f"{day_word} {month_name} {self._year_to_words(year)}"
            else:
                return f"{month_name} {day_word}, {self._year_to_words(year)}"

        pattern = r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
        return re.sub(pattern, replace_date, text)

    def _year_to_words(self, year: str) -> str:
        """Convertit une annee en mots."""
        y = int(year)
        if 1000 <= y <= 1999:
            # Ex: 1984 -> "mille neuf cent quatre-vingt-quatre"
            return self.number_converter.convert(y)
        elif 2000 <= y <= 2099:
            if self.lang == "fr":
                if y == 2000:
                    return "deux mille"
                return "deux mille " + self.number_converter.convert(y - 2000)
            else:
                if y == 2000:
                    return "two thousand"
                return "two thousand " + self.number_converter.convert(y - 2000)
        return self.number_converter.convert(y)

    def _normalize_times(self, text: str) -> str:
        """Convertit les heures en texte."""
        def replace_time(match):
            hour = int(match.group(1))
            minute = int(match.group(2))

            hour_word = self.number_converter.convert(hour)

            if self.lang == "fr":
                heure = "heure" if hour <= 1 else "heures"
                if minute == 0:
                    return f"{hour_word} {heure}"
                minute_word = self.number_converter.convert(minute)
                return f"{hour_word} {heure} {minute_word}"
            else:
                if minute == 0:
                    return f"{hour_word} o'clock"
                minute_word = self.number_converter.convert(minute)
                return f"{hour_word} {minute_word}"

        # Format HH:MM ou HHhMM
        pattern = r'(\d{1,2})[:hH](\d{2})'
        return re.sub(pattern, replace_time, text)

    def _normalize_phone_numbers(self, text: str) -> str:
        """Normalise les numeros de telephone."""
        def replace_phone(match):
            number = match.group(0)
            # Separer par groupes de 2 chiffres
            digits = re.sub(r'[^\d]', '', number)
            groups = [digits[i:i+2] for i in range(0, len(digits), 2)]
            words = [self.number_converter.convert(int(g)) for g in groups if g]
            return " ".join(words)

        # Patterns de telephone
        patterns = [
            r'\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}',  # FR
            r'\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}',  # International
        ]

        for pattern in patterns:
            text = re.sub(pattern, replace_phone, text)
        return text

    def _normalize_roman_numerals(self, text: str) -> str:
        """Convertit les chiffres romains en mots."""
        def roman_to_int(s: str) -> int:
            total = 0
            prev = 0
            for char in reversed(s):
                value = self.ROMAN_VALUES.get(char, 0)
                if value < prev:
                    total -= value
                else:
                    total += value
                prev = value
            return total

        def replace_roman(match):
            roman = match.group(1)
            # Verifier que c'est vraiment un chiffre romain
            if not all(c in self.ROMAN_VALUES for c in roman):
                return match.group(0)
            number = roman_to_int(roman)
            if number > 0:
                return self.number_converter.convert(number)
            return match.group(0)

        # Pattern pour chiffres romains (avec contexte)
        # Ex: "chapitre III", "Louis XIV", "XXe siecle"
        pattern = r'\b([IVXLCDM]+)\b'
        return re.sub(pattern, replace_roman, text)

    def _normalize_numbers_with_units(self, text: str) -> str:
        """Convertit les nombres avec unites."""
        def replace_with_unit(match):
            number = match.group(1)
            unit = match.group(2)

            # Convertir le nombre (gerer les decimales)
            if ',' in number or '.' in number:
                number = number.replace(',', '.')
                parts = number.split('.')
                integer_part = self.number_converter.convert(int(parts[0]))
                if self.lang == "fr":
                    decimal_word = "virgule"
                else:
                    decimal_word = "point"
                decimal_part = " ".join(
                    self.number_converter.convert(int(d)) for d in parts[1]
                )
                number_word = f"{integer_part} {decimal_word} {decimal_part}"
            else:
                number_word = self.number_converter.convert(int(number))

            # Convertir l'unite
            unit_word = self.units.get(unit, unit)

            return f"{number_word} {unit_word}"

        # Pattern: nombre + unite
        units_pattern = '|'.join(re.escape(u) for u in self.units.keys())
        pattern = rf'(\d+(?:[.,]\d+)?)\s*({units_pattern})\b'
        return re.sub(pattern, replace_with_unit, text, flags=re.IGNORECASE)

    def _normalize_percentages(self, text: str) -> str:
        """Convertit les pourcentages."""
        def replace_percent(match):
            number = match.group(1).replace(',', '.')
            if '.' in number:
                parts = number.split('.')
                integer_part = self.number_converter.convert(int(parts[0]))
                if self.lang == "fr":
                    return f"{integer_part} virgule {parts[1]} pourcent"
                return f"{integer_part} point {parts[1]} percent"
            else:
                word = self.number_converter.convert(int(number))
                pct = "pourcent" if self.lang == "fr" else "percent"
                return f"{word} {pct}"

        pattern = r'(\d+(?:[.,]\d+)?)\s*%'
        return re.sub(pattern, replace_percent, text)

    def _normalize_currency(self, text: str) -> str:
        """Convertit les montants monetaires."""
        currencies = {
            "€": ("euros", "euro"),
            "$": ("dollars", "dollar"),
            "£": ("livres", "pounds" if self.lang == "en" else "livres sterling"),
            "¥": ("yens", "yen"),
        }

        def replace_currency(match):
            symbol = match.group(1) or match.group(3)
            amount = match.group(2).replace(',', '.').replace(' ', '')

            currency_name = currencies.get(symbol, (symbol, symbol))

            if '.' in amount:
                parts = amount.split('.')
                integer_part = self.number_converter.convert(int(parts[0]))
                cents = int(parts[1].ljust(2, '0')[:2])
                if cents > 0:
                    cents_word = self.number_converter.convert(cents)
                    if self.lang == "fr":
                        return f"{integer_part} {currency_name[0]} et {cents_word} centimes"
                    return f"{integer_part} {currency_name[1]} and {cents_word} cents"
                return f"{integer_part} {currency_name[0]}"
            else:
                word = self.number_converter.convert(int(amount))
                name = currency_name[1] if int(amount) == 1 else currency_name[0]
                return f"{word} {name}"

        # Patterns: €10, 10€, $10, 10$
        pattern = r'([€$£¥])\s?(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s?([€$£¥])'
        return re.sub(pattern, replace_currency, text)

    def _normalize_ordinals(self, text: str) -> str:
        """Convertit les ordinaux (1er, 2eme, 3rd, etc.)."""
        if self.lang == "fr":
            def replace_ordinal(match):
                number = int(match.group(1))
                if number == 1:
                    return "premier"
                word = self.number_converter.convert(number)
                return word + "ieme"

            pattern = r'(\d+)(?:er|ere|eme|ème|e)\b'
        else:
            def replace_ordinal(match):
                number = int(match.group(1))
                word = self.number_converter.convert(number)
                if number == 1:
                    return "first"
                elif number == 2:
                    return "second"
                elif number == 3:
                    return "third"
                return word + "th"

            pattern = r'(\d+)(?:st|nd|rd|th)\b'

        return re.sub(pattern, replace_ordinal, text, flags=re.IGNORECASE)

    def _normalize_numbers(self, text: str) -> str:
        """Convertit les nombres restants."""
        def replace_number(match):
            number_str = match.group(0)

            # Gerer les decimales
            if ',' in number_str or '.' in number_str:
                number_str = number_str.replace(',', '.').replace(' ', '')
                parts = number_str.split('.')
                try:
                    integer_part = self.number_converter.convert(int(parts[0]))
                    if self.lang == "fr":
                        decimal_word = "virgule"
                    else:
                        decimal_word = "point"
                    decimal_part = " ".join(
                        self.number_converter.convert(int(d)) for d in parts[1]
                    )
                    return f"{integer_part} {decimal_word} {decimal_part}"
                except ValueError:
                    return match.group(0)
            else:
                try:
                    number = int(number_str.replace(' ', ''))
                    return self.number_converter.convert(number)
                except ValueError:
                    return match.group(0)

        # Nombres avec espaces comme separateurs de milliers
        pattern = r'\d{1,3}(?:[\s,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?'
        return re.sub(pattern, replace_number, text)

    def _normalize_abbreviations(self, text: str) -> str:
        """Expanse les abreviations."""
        for abbr, expansion in self.abbreviations.items():
            # Avec ou sans point final
            pattern = r'\b' + re.escape(abbr) + r'(?=\s|$|[,;:])'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text

    def _normalize_symbols(self, text: str) -> str:
        """Remplace les symboles par leur equivalent verbal."""
        for symbol, replacement in self.symbols.items():
            text = text.replace(symbol, replacement)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalise les espaces."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def normalize_text_for_tts(text: str, lang: str = "fr") -> str:
    """
    Fonction utilitaire pour normaliser du texte.

    Args:
        text: Texte a normaliser
        lang: Code langue

    Returns:
        Texte normalise pret pour TTS
    """
    normalizer = TextNormalizer(lang)
    return normalizer.normalize(text)


if __name__ == "__main__":
    # Tests
    test_texts = [
        "Le 25/12/2024, il etait 14h30.",
        "Le prix est de 1 234,56 € soit environ 1500$.",
        "Mon tel: 06 12 34 56 78",
        "Chapitre III: Louis XIV au XVIIe siecle",
        "Il a couru 42,195 km en 3h45min.",
        "Environ 85% des participants...",
        "Voir https://example.com pour plus d'infos.",
        "Contact: info@example.com",
        "M. Dupont et Mme Martin, Dr. House, etc.",
        "La temperature est de 23°C.",
        "C'est le 1er janvier 2000.",
    ]

    print("=== Test normalisation francais ===\n")
    normalizer = TextNormalizer("fr")

    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"Avant: {text}")
        print(f"Apres: {normalized}")
        print()
