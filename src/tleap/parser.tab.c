/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 74 "parser.y"

#include        <unistd.h>
#include        "basics.h"

#include        "classes.h"

#include        "dictionary.h"
#include        "parmLib.h"

#include        "commands.h"
#include	"block.h"
#include	"parser.h"

#include        "leap.h"

#include        "help.h"

int yyerror( char *sStr );
int yylex();

#define         MESSAGEFILTER   MESSPARSER




#define         NULLSTR         "null"

/*
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        GLOBAL VARIABLES

*/

                /* The global variable GfCurrentInput defines the file */
                /* from which input is currently read.  When the file */
                /* is empty, then switch back to stdin */

#define	MAXINPUT	1000
#define	MAXINPUTFILES	10		/* Maximum 10 input files */
					/* can be open at once */


char            GsInputLine[MAXINPUT] = "";
BOOL		GbLastLine = FALSE;
BOOL		bCmdDeleteObj;
int             GiInputPos = 0;
PARMLIB		GplAllParameters;
RESULTt		GrMainResult;
BLOCK		GbCommand = NULL;
BLOCK		GbExecute = NULL;
int		GiClipPrompts = 0;
BOOL		GbGraphicalEnvironment;
STRING		GsProgramName;

extern int	iMemDebug;

static	STRING	*SbFirstSourceFiles = NULL;
static	int	iFirstSource = 0;
static	BOOL	SbUseStartup = TRUE;



/*
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *
 *	The following is used by the parser to maintain a stack of
 *	files where input is received from.  If the file is NULL
 *	then input is received from the main program in the form
 *	of BLOCKS.  The main program is then responsible for reading
 *	the stdin ( in the command line interface ) or for gathering
 *	keypress events from X-Windows ( in the graphical interface ).
 *
 */


int             GiInputFileStackPos = 0;
FILE*           GfaInputFileStack[MAXINPUTFILES];



/*
 *----------------------------------------------------------------
 *
 *	Not quite GLOBAL variables used by the parser.
 */

                                /* Arguments to routines are passed through */
                                /* an array */
#define MAXARGS         10
#define MAXLISTNEXT     10


ATOM            aDummy;
ASSOC           aaArgs[MAXARGS];
int             iArgCount, i;

                                /* List stuff is used for input of nested */
                                /* lists */
#define MAXLISTNEST     10
ASSOC           aaLists[MAXLISTNEST];
int             iCurrentList = -1;
#define PUSHLIST()      iCurrentList++
#define POPLIST()       iCurrentList--
#define CURRENTLIST     aaLists[iCurrentList]


OBJEKT          o0;
double          dTemp;
ASSOC           aAssoc;
STRING          sTemp;
BOOL		bQuit = FALSE;
BOOL		bCommandFound = FALSE;

                /* There seems to be a problem with YACC not properly */
                /* declaring yylval and yyval */
typedef struct  {
	ASSOC		aVal;
	double		dVal;
	STRING		sVal;
	FUNCTION	fCallback;
} YYSTYPEt;

#define YYSTYPE YYSTYPEt


extern  OBJEKT  oGetObject();           /* ( STRING ) */

#line 200 "parser.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif


/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    LVARIABLE = 258,               /* LVARIABLE  */
    LSTRING = 259,                 /* LSTRING  */
    LNUMBER = 260,                 /* LNUMBER  */
    LASSIGN = 261,                 /* LASSIGN  */
    LENDOFCOMMAND = 262,           /* LENDOFCOMMAND  */
    LOPENLIST = 263,               /* LOPENLIST  */
    LCLOSELIST = 264,              /* LCLOSELIST  */
    LOPENPAREN = 265,              /* LOPENPAREN  */
    LCLOSEPAREN = 266,             /* LCLOSEPAREN  */
    LQUIT = 267,                   /* LQUIT  */
    LCOMMA = 268,                  /* LCOMMA  */
    LDOT = 269,                    /* LDOT  */
    LCOMMAND = 270,                /* LCOMMAND  */
    LDUMMY = 271,                  /* LDUMMY  */
    LNULL = 272,                   /* LNULL  */
    LNOTSINGLECHAR = 273           /* LNOTSINGLECHAR  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */


extern YYSTYPE yylval;


int yyparse (void);



/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_LVARIABLE = 3,                  /* LVARIABLE  */
  YYSYMBOL_LSTRING = 4,                    /* LSTRING  */
  YYSYMBOL_LNUMBER = 5,                    /* LNUMBER  */
  YYSYMBOL_LASSIGN = 6,                    /* LASSIGN  */
  YYSYMBOL_LENDOFCOMMAND = 7,              /* LENDOFCOMMAND  */
  YYSYMBOL_LOPENLIST = 8,                  /* LOPENLIST  */
  YYSYMBOL_LCLOSELIST = 9,                 /* LCLOSELIST  */
  YYSYMBOL_LOPENPAREN = 10,                /* LOPENPAREN  */
  YYSYMBOL_LCLOSEPAREN = 11,               /* LCLOSEPAREN  */
  YYSYMBOL_LQUIT = 12,                     /* LQUIT  */
  YYSYMBOL_LCOMMA = 13,                    /* LCOMMA  */
  YYSYMBOL_LDOT = 14,                      /* LDOT  */
  YYSYMBOL_LCOMMAND = 15,                  /* LCOMMAND  */
  YYSYMBOL_LDUMMY = 16,                    /* LDUMMY  */
  YYSYMBOL_LNULL = 17,                     /* LNULL  */
  YYSYMBOL_LNOTSINGLECHAR = 18,            /* LNOTSINGLECHAR  */
  YYSYMBOL_YYACCEPT = 19,                  /* $accept  */
  YYSYMBOL_input = 20,                     /* input  */
  YYSYMBOL_line = 21,                      /* line  */
  YYSYMBOL_instruct = 22,                  /* instruct  */
  YYSYMBOL_assign = 23,                    /* assign  */
  YYSYMBOL_express = 24,                   /* express  */
  YYSYMBOL_rawexp = 25,                    /* rawexp  */
  YYSYMBOL_26_1 = 26,                      /* $@1  */
  YYSYMBOL_function = 27,                  /* function  */
  YYSYMBOL_28_2 = 28,                      /* $@2  */
  YYSYMBOL_elements = 29,                  /* elements  */
  YYSYMBOL_cmdname = 30,                   /* cmdname  */
  YYSYMBOL_args = 31,                      /* args  */
  YYSYMBOL_arg = 32                        /* arg  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  13
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   46

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  19
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  14
/* YYNRULES -- Number of rules.  */
#define YYNRULES  27
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  34

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   273


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   224,   224,   230,   231,   235,   242,   243,   246,   264,
     271,   272,   276,   275,   289,   298,   308,   316,   323,   335,
     334,   367,   368,   378,   381,   382,   383,   386
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "LVARIABLE", "LSTRING",
  "LNUMBER", "LASSIGN", "LENDOFCOMMAND", "LOPENLIST", "LCLOSELIST",
  "LOPENPAREN", "LCLOSEPAREN", "LQUIT", "LCOMMA", "LDOT", "LCOMMAND",
  "LDUMMY", "LNULL", "LNOTSINGLECHAR", "$accept", "input", "line",
  "instruct", "assign", "express", "rawexp", "$@1", "function", "$@2",
  "elements", "cmdname", "args", "arg", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-18)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int8 yypact[] =
{
      31,    -3,    -1,   -18,   -18,     7,   -18,   -18,     1,     2,
     -18,   -18,    -2,   -18,   -18,   -18,    20,   -18,   -18,   -18,
     -18,   -18,   -18,   -18,   -18,   -18,   -18,    20,   -18,   -18,
     -18,    13,   -18,   -18
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     3,    23,     0,     2,     4,     0,     0,
      19,     5,     9,     1,     6,     7,    24,    16,    15,    14,
      12,    17,    18,     8,    10,    11,    27,    20,    25,    21,
      26,     0,    13,    22
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -18,   -18,   -18,   -18,   -18,   -18,   -12,   -18,     0,   -18,
     -18,   -18,   -18,   -17
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
       0,     5,     6,     7,     8,    23,    26,    29,     9,    16,
      31,    10,    27,    28
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      24,    17,    18,    19,    11,    12,    20,    13,    14,    15,
      30,     0,    25,     4,    21,    22,    17,    18,    19,    33,
       0,    20,    32,    17,    18,    19,     0,     0,    20,    21,
      22,     0,     1,     0,     2,     0,    21,    22,     3,     0,
       0,     0,     0,     0,     0,     0,     4
};

static const yytype_int8 yycheck[] =
{
      12,     3,     4,     5,     7,     6,     8,     0,     7,     7,
      27,    -1,    12,    15,    16,    17,     3,     4,     5,    31,
      -1,     8,     9,     3,     4,     5,    -1,    -1,     8,    16,
      17,    -1,     1,    -1,     3,    -1,    16,    17,     7,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    15
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     1,     3,     7,    15,    20,    21,    22,    23,    27,
      30,     7,     6,     0,     7,     7,    28,     3,     4,     5,
       8,    16,    17,    24,    25,    27,    25,    31,    32,    26,
      32,    29,     9,    25
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    19,    20,    21,    21,    21,    22,    22,    23,    23,
      24,    24,    26,    25,    25,    25,    25,    25,    25,    28,
      27,    29,    29,    30,    31,    31,    31,    32
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     1,     2,     2,     2,     3,     2,
       1,     1,     0,     4,     1,     1,     1,     1,     1,     0,
       3,     0,     2,     1,     0,     1,     2,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* input: line  */
#line 225 "parser.y"
                {
			return 0;
		}
#line 1285 "parser.tab.c"
    break;

  case 4: /* line: instruct  */
#line 232 "parser.y"
                        {
                        bCommandFound = FALSE;
                        }
#line 1293 "parser.tab.c"
    break;

  case 5: /* line: error LENDOFCOMMAND  */
#line 236 "parser.y"
                        {
                            VP0(( "\n" ));
                            yyerrok;
                        }
#line 1302 "parser.tab.c"
    break;

  case 8: /* assign: LVARIABLE LASSIGN express  */
#line 247 "parser.y"
                        {
                                /* Set the value of the variable */
                            aAssoc = (yyvsp[0].aVal);
			    if ( aAssoc != NULL ) {
                                MESSAGE(( "Assigning a value to %s\n", 
                                                        (yyvsp[-2].sVal) ));
                                VariableSet( (yyvsp[-2].sVal), oAssocObject(aAssoc) );
				MESSAGE(( "DEREF (assign) - %s\n",
							sAssocName(aAssoc) ));
                                DEREF( aAssoc );
			    } else {
				MESSAGE(("Not assigning value to %s - rmving\n",
							(yyvsp[-2].sVal) ));
				VariableRemove( (yyvsp[-2].sVal) );
			    }

                        }
#line 1324 "parser.tab.c"
    break;

  case 9: /* assign: LVARIABLE LASSIGN  */
#line 265 "parser.y"
                        {
                            MESSAGE(( "Removing variable %s\n", (yyvsp[-1].sVal) ));
                            VariableRemove( (yyvsp[-1].sVal) );
                        }
#line 1333 "parser.tab.c"
    break;

  case 12: /* $@1: %empty  */
#line 276 "parser.y"
                        {
                            PUSHLIST();
                                /* Create an ASSOC for the list */
                            aAssoc = (ASSOC)oCreate(ASSOCid);
                            AssocSetName( aAssoc, "" );
                            AssocSetObject( aAssoc, oCreate(LISTid) );
                            CURRENTLIST = aAssoc;
                        }
#line 1346 "parser.tab.c"
    break;

  case 13: /* rawexp: LOPENLIST $@1 elements LCLOSELIST  */
#line 285 "parser.y"
                        {
                            (yyval.aVal) = (yyvsp[-1].aVal);
                            POPLIST();
                        }
#line 1355 "parser.tab.c"
    break;

  case 14: /* rawexp: LNUMBER  */
#line 290 "parser.y"
                        {
                            aAssoc = (ASSOC)oCreate(ASSOCid);
                            AssocSetName( aAssoc, "" );
                            o0 = oCreate(ODOUBLEid);
                            ODoubleSet( o0, (yyvsp[0].dVal) );
                            AssocSetObject( aAssoc, o0 );
                            (yyval.aVal) = aAssoc;
                        }
#line 1368 "parser.tab.c"
    break;

  case 15: /* rawexp: LSTRING  */
#line 299 "parser.y"
                        {
                            aAssoc = (ASSOC)oCreate(ASSOCid);
                            AssocSetName( aAssoc, "" );
                            o0 = oCreate(OSTRINGid);
                            OStringDefine( (OSTRING) o0, (yyvsp[0].sVal) );
                            AssocSetObject( aAssoc, o0 );
			    DEREF( o0 );	/* keeps count = 1 */
                            (yyval.aVal) = aAssoc;
                        }
#line 1382 "parser.tab.c"
    break;

  case 16: /* rawexp: LVARIABLE  */
#line 309 "parser.y"
                        {
			    OBJEKT	o = oGetObject((yyvsp[0].sVal));
                            aAssoc = (ASSOC)oCreate(ASSOCid);
                            AssocSetName( aAssoc, (yyvsp[0].sVal) );
                            AssocSetObject( aAssoc, o ); /* REF's o */
                            (yyval.aVal) = aAssoc;
                        }
#line 1394 "parser.tab.c"
    break;

  case 17: /* rawexp: LDUMMY  */
#line 317 "parser.y"
                        {
                            aAssoc = (ASSOC)oCreate(ASSOCid);
                            AssocSetName( aAssoc, "" );
                            AssocSetObject( aAssoc, aDummy );
                            (yyval.aVal) = aAssoc;
                        }
#line 1405 "parser.tab.c"
    break;

  case 18: /* rawexp: LNULL  */
#line 324 "parser.y"
                        {
                            MESSAGE(( "Parsed a null\n" ));
                            aAssoc = (ASSOC)oCreate(ASSOCid);
                            AssocSetName( aAssoc, "" );
                            AssocSetObject( aAssoc, NULL );
                            (yyval.aVal) = aAssoc;
                        }
#line 1417 "parser.tab.c"
    break;

  case 19: /* $@2: %empty  */
#line 335 "parser.y"
                        { iArgCount = 0;
                        }
#line 1424 "parser.tab.c"
    break;

  case 20: /* function: cmdname $@2 args  */
#line 338 "parser.y"
                        {
                                /* Execute the command */
			    MESSAGE(( "executing function\n"));
			    bCmdDeleteObj = FALSE;
                            o0 = (yyvsp[-2].fCallback)( iArgCount, aaArgs );
			    if ( o0 != NULL ) {
                            	aAssoc = (ASSOC)oCreate(ASSOCid);
                            	AssocSetObject( aAssoc, o0 );
                            	(yyval.aVal) = aAssoc;
			    } else {
				MESSAGE(( "func == NULL---\n"));
				(yyval.aVal) = NULL;
			    }

                                /* DEREF each of the arguments */

                            for ( i=0; i<iArgCount; i++ ) {
				if ( bCmdDeleteObj ) {
					MESSAGE(( "bCmdDeleteObj---\n" ));
					DEREF( oAssocObject( aaArgs[i] ) );
				}
				MESSAGE(( "DEREF (function) - %s\n",
						sAssocName(aaArgs[i])));
                                DEREF( aaArgs[i] );
                            }
                        }
#line 1455 "parser.tab.c"
    break;

  case 22: /* elements: elements rawexp  */
#line 369 "parser.y"
                        {
                                /* Get the element and add it to the list */
                            MESSAGE(( "Adding to list:\n" ));
                            ListAddToEnd( (LIST)oAssocObject(CURRENTLIST), 
							(OBJEKT) (yyvsp[0].aVal) );
                            (yyval.aVal) = CURRENTLIST;
                        }
#line 1467 "parser.tab.c"
    break;

  case 27: /* arg: rawexp  */
#line 387 "parser.y"
                        {
                            aaArgs[iArgCount++] = (yyvsp[0].aVal);
                        }
#line 1475 "parser.tab.c"
    break;


#line 1479 "parser.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 403 "parser.y"

/*------------------------------------------------------------

        ROUTINES

*/

static  BOOL    SbGotUngetc = FALSE;
static  char    ScUngetc;


/*
 *      yyerror
 *
 *      Respond to errors.
 */
int
yyerror( char *sStr )
{
    VP0(( "ERROR: %s\n", sStr ));
}

FILE *
fINPUTFILE()            
{
	if ( GiInputFileStackPos < 0 )
		return(NULL);
	return( GfaInputFileStack[GiInputFileStackPos] );
}



/*
 *	zbGetLine
 *
 *	Get the next line of input from the current input source.
 *	This may be GbCommand or GbExecute depending if
 *	the input is coming from the user or from an execute file.
 *	Return FALSE if there is no more lines to be received from 
 *	the BLOCK.
 */
BOOL
zbGetLine( char *sLine, BOOL *bPFromExecute )
{
BOOL		bGotBlock;
char		c;

    if ( fINPUTFILE() == NULL ) {
	*bPFromExecute = FALSE;
	return(bBlockReadLine( GbCommand, sLine ));
    }

    *bPFromExecute = TRUE;

		/* If there is no line in the execute BLOCK */
		/* then read in another block from the current file */

    if ( bBlockEndOfRead(GbExecute) ) {
	    BlockEmpty( GbExecute );
	    bGotBlock = FALSE;
	    while ( !feof(fINPUTFILE()) && !bGotBlock ) {
		c = fgetc(fINPUTFILE());
		if ( feof(fINPUTFILE()) ) break;
		bGotBlock = bBlockAddChar( GbExecute, c );
	    }

		/* If a complete BLOCK was not read then */
		/* append a final '\n' character, if that doesn't */
		/* make this a complete BLOCK then there is an error */
		/* in the input, also we are at the end of the file */
		/* so pop the file off the stack and say that we have */
		/* a complete BLOCK */

	    if ( !bGotBlock ) {
	    	bGotBlock = bBlockAddChar( GbExecute, '\n' );
	    	if ( fINPUTFILE() != NULL )
	    		fclose( fINPUTFILE() );
	    	INPUTPOPFILE();
	    }
    }
    return(bBlockReadLine( GbExecute, sLine ));
}


		
	


/*
 *      zcGetChar
 *
 *      Get the next character from the line buffer.
 *	If there are no more characters in the line buffer and
 *	GbLastLine is TRUE then return '\0', otherwise if the
 *	line buffer is empty, fill it and return the next character
 *	in it.
 */
char
zcGetChar()
{
char            c;
BOOL		bFromExecute;

                /* If there is a pushed character then return it */

    if ( SbGotUngetc ) {
        SbGotUngetc = FALSE;
        c = ScUngetc;
        goto DONE;
    }

		/* Now if the input line is empty then fill it */
    if ( GsInputLine[GiInputPos] == '\0' ) {
	if ( GbLastLine ) {
	    c = '\0';
	    GbLastLine = FALSE;
	    goto DONE;
	}
	GbLastLine = zbGetLine( GsInputLine, &bFromExecute );
	GiInputPos = 0;
	if ( bFromExecute ) {
	    for ( i=GiClipPrompts; i<=iINPUTSTACKDEPTH(); i++ ) VP2(( ">" ));
	    VP2(( " " ));
	    VP2(( "%s", GsInputLine ));
	} else {
	    VPLOG(( "> %s", GsInputLine ));
	}
    }

    c = GsInputLine[GiInputPos++];

DONE:
    return(c);

}



/*
 *      zUngetc
 *
 *      Push one character back to the file.
 */
void
zUngetc( char c )
{
    SbGotUngetc = TRUE;
    ScUngetc = c;
}






/*
 *      cGetChar
 *
 *      Return the next character, skipping over comments
 *      which start with '#' and end with '\n'.
 */
char
cGetChar()
{
char    c;

	while (1) {
        	c = zcGetChar();
        	if ( c == '#' )
            		while ( ( c=zcGetChar() ) != '\n' )  /* Nothing */ ;
		else
			break;
        } 
	return(c);
}

 
int
iOneCharToken( int c )
{
	switch(c) {
		case ';':
        		return(LENDOFCOMMAND);
		case '=':
        		return(LASSIGN);
		case '*':
        		return(LDUMMY);
		case '(':
        		return(LOPENPAREN);
		case ')':
        		return(LCLOSEPAREN);
		case '{':
        		return(LOPENLIST);
		case '}':
        		return(LCLOSELIST);
		default:
        		return(LNOTSINGLECHAR);
	}
}

/*
 *      yylex
 *
 *      Lexical analyzer for the parser.
 *      Read characters from stdin and return the token types
 *      read and place the value read into the global UNION
 *      yylval.
 *
 *      The things that it recognizes are:
 *              LDOUBLE         [-]###.###E## or ###
 *              LSTRING         "xx xx xxx" or '$'everything up to ' ' ',' ';'
 *              commands        xxxxxxx
 *              LVARIABLE       xxxxxxx which are not commands
 *              LTERMINATOR     ;
 *              LASSIGN         =
 *
 *	Modified 17 November 1992 - David A. Rivkin
 *		Added checking the alias table for command matches.
 *	Total rewrite October 1993 - Bill Ross
 *
 */
int
yylex()
{
STRING          sStr;
int             j, iMax, tok;
BOOL            bGotExp, bGotDot;
char            c;
STRING		sCmd;
STRING		sPossibleCmd;

                /* Skip over blanks, tabs, end of lines etc */

    while ( (c=cGetChar())==' '  ||  c=='\t'  ||  c=='\n'  ||  c == ',' );

    if ( c == '\0' ) 
	return(LENDOFCOMMAND);

    /*
     *  Check the 1-character possibilities: , ; = * ( ) { }
     */
    tok = iOneCharToken( c );
    if ( tok != LNOTSINGLECHAR ) {
        MESSAGE(( "Parsed /%c/\n", c ));
	return(tok);
    }

    /*
     *  it isn't a 1-char thing; read in the rest 
     *	and push back the terminating char
     */
    sStr[0] = c;
    for (j=1;;j++) {

	if ( j >= sizeof(STRING) )
	    DFATAL(( "string too long" ));

	c = cGetChar();
	/*
	 *  NULL terminates anything (?)
	 */
	if ( c == '\0' ) {
	    	sStr[j] = '\0';
	    	break;
	}
	/*
	 *  allow anything inside quotes; chop closing quote
	 */
	if ( sStr[0] == '"' ) {
		if ( c == '"' ) {
			sStr[j] = '\0';
			break;
		}
		sStr[j] = c;
		continue;
	}
	/*
	 *  whitespace is a delimiter outside of quotes
	 */
	if ( c == ' '  ||  c == '\t'  ||  c == '\n'  ||  c == ',' ) {
	    	sStr[j] = '\0';
	    	break;
	}
	/*
	 *  special case for $-type strings: allow embedded single-char
	 *	tokens, except ';'
	 */
	if ( sStr[0] == '$' ) {
		if ( c == ';' ) {
			zUngetc( c );
	    		sStr[j] = '\0';
	    		break;
		}
		sStr[j] = c;
		continue;
	}
	tok = iOneCharToken( c );
	if ( tok != LNOTSINGLECHAR ) {
		zUngetc( c );
	    	sStr[j] = '\0';
	    	break;
	}
	sStr[j] = c;
    }

    /*
     *  see if it's a number
     */
    bGotExp = FALSE;
    bGotDot = FALSE;
    if ( isdigit(sStr[0]) || sStr[0] == '-' || sStr[0] == '+' || 
				( sStr[0] == '.' && isdigit(sStr[1]) ) ) {

        for ( j=0; j<sizeof(STRING); j++ ) {
            MESSAGE(( "Thinking NUMBER got: %c\n", sStr[j] ));
	    switch ( sStr[j] ) {
		case '\0':
        	    if ( sscanf( sStr, "%lf", &yylval.dVal ) != 1 ) {
			VP0(( " Couldn't scan NUMBER from (%s)\n", sStr ));
			return(LNULL);
		    }
        	    MESSAGE(( "Parsed a number: %lf\n", yylval.dVal ));
        	    return(LNUMBER);
		case '.':
		    if ( bGotDot ) {
			VP0(( "(Multiple '.' in NUMBER-like thing (%s))\n", 
								sStr ));
			goto notnum;
		    }
		    if ( bGotExp ) {
			VP0(( 
			 "('.' follows exponent in NUMBER-like thing (%s))\n",
								sStr ));
			goto notnum;
		    }
        	    bGotDot = TRUE;
		    break;
		case 'e':
		case 'E':
            	    if ( bGotExp ) {
			VP0(( "(Multiple 'e' in NUMBER-like thing (%s))\n", 
								sStr ));
			goto notnum;
		    }
                    bGotExp = TRUE;
                    break;
		case '+':
		case '-':
                    break;
		default:
            	    if ( !isdigit(sStr[j]) ) {
			goto notnum;
		    }
            	    break;
	    }
	}
    }
notnum:
    /* 
     *  see if it's a string in quotes
     */
    if ( sStr[0] == '"' ) {
        strcpy( yylval.sVal, &sStr[1] );
        MESSAGE(( "Parsed a STRING: %s\n", sStr ));
        return(LSTRING);
    }

    /* 
     *  see if it's a string prefixed w/ '$'
     */
    if ( sStr[0] == '$' ) {
        strcpy( yylval.sVal, &sStr[1] );
        MESSAGE(( "Parsed a STRING: %s\n", sStr ));
        return(LSTRING);
    }

                /* LASTLY!!!!!!!! */
    /* 
     *  see if it's a variable/command 
     */
    strcpy( yylval.sVal, sStr );
    strcpy( sPossibleCmd, sStr );
    StringLower( sPossibleCmd );

    		/* Check if there is an alias that is an exact match */
    if ( iMax = iVarArrayElementCount( GvaAlias ) ) {
	ALIAS		aAlias;
	aAlias = PVAI( GvaAlias, ALIASt, 0 );
	for ( i=0; i<iMax; i++, aAlias++ ) {
	    if ( strcmp( aAlias->sName, sPossibleCmd ) == 0 ) {
	    	strcpy( sPossibleCmd, aAlias->sCommand );
	    }
        }
    }
                /* Check if there is an exact match of the command */
                /* If a command has already been found for this input
                	line, then do not consider the string a command
                	but rather as a STRING variable */
                	
    if ( !bCommandFound ) {
	for ( j=0; strlen(cCommands[j].sName) != 0; j++ ) {
	    strcpy( sCmd, cCommands[j].sName );
	    StringLower( sCmd );
            if ( strcmp( sCmd, sPossibleCmd ) == 0 ) {
		yylval.fCallback = cCommands[j].fCallback;
		MESSAGE(( "Parsed a command: %s\n", sStr ));
		bCommandFound = TRUE;
		return(LCOMMAND);
	    }
        }
    }


                /* If the variable name is null then return LNULL */

    if ( strcmp( sStr, NULLSTR ) == 0 ) 
	return(LNULL);
    
                /* Return the variable name */

    strcpy( yylval.sVal, sStr );
    MESSAGE(( "Parsed a variable: %s\n", sStr ));
    return(LVARIABLE);

}




/*
 *      oGetObject
 *
 *      If the string is a variable then return the OBJEKT that
 *      it is attached to, otherwise if the string is
 *      a string like: 'unit.mol.res.atom' then parse the
 *      individual names and search the CONTAINERS for
 *      the subcontainers.
 */
OBJEKT
oGetObject( char *sName )
{
CONTAINER       cCont[5];
int             j, k, iSeq;
OBJEKT          oObj;
STRING          sLine, sHead;
BOOL		bDot, bAt, bPdbSeq;
STRING		sGroup;
LIST		lGroup;
OSTRING		osString;
LOOP		lRes;
RESIDUE		rRes;

    oObj = oVariable( sName );
    if ( oObj != NULL ) return(oObj);

        /* Now try to parse the name */
    strcpy( sLine, sName );
    bDot = FALSE;
    bAt = FALSE;
    bPdbSeq = FALSE;
    for ( k=0; k<strlen(sName); k++ ) {
	if ( sLine[k] == '.' ) {
	    sLine[k]=' ';
	    bDot = TRUE;
/*fprintf(stderr, "GOTDOT\n"); */
	} else if ( sLine[k] == '@' ) {
	    sLine[k] = ' ';
	    bAt = TRUE;
	} else if ( sLine[k] == '%' ) {
	    if ( !bDot ) {
	        sLine[k] = ' ';
	        bPdbSeq = TRUE;
	    }
	}
    }


    sRemoveFirstString( sLine, sHead );
    cCont[0] = (CONTAINER)oVariable(sHead);
    if ( cCont[0] == NULL ) {
/* fprintf(stderr, "STRING %s\n", sName); */
    	/* It is not an object variable so...
    	   return the whole string as a OSTRING */
	goto String;
    }
    
/*fprintf(stderr, "VARIABLE %c %c\n", bAt, bPdbSeq);*/
    if ( bPdbSeq ) {
	sRemoveLeadingSpaces( sLine );
	sRemoveFirstString( sLine, sHead );
	if ( strlen(sHead) == 0 ) {
	    goto String;
	}
	if ( isdigit(sHead[0]) ) {

			/* Make sure the rest are digits */
			/* If not return NULL */
	    for ( j=1; j<strlen(sHead); j++ ) {
		if ( !isdigit(sHead[j]) ) {
			    /* It is not an object variable so...
			       return the whole string as a OSTRING */
		    goto String;
		}
	    }
	
			/* Find the PDB sequence number */

	    cCont[1] = NULL;	
	    iSeq = atoi(sHead);
	    lRes = lLoop( (OBJEKT)cCont[0], RESIDUES );
	    while ( rRes = (RESIDUE)oNext(&lRes) ) {
		if ( iResiduePdbSequence(rRes)==iSeq ) {
		    cCont[1] = (CONTAINER) rRes;
		}
	    }

	    if ( !cCont[1] ) {
		goto String;
	    }

	    if ( bDot ) {
		sRemoveLeadingSpaces( sLine );
		sRemoveFirstString( sLine, sHead );
		if ( strlen(sHead) == 0 ) {
		    goto String;
		}
		if ( isdigit(sHead[0]) ) {

				/* Make sure the rest are digits */
				/* If not return NULL */
		    for ( j=1; j<strlen(sHead); j++ ) {
			if ( !isdigit(sHead[j]) ) {
				    /* It is not an object variable so...
				       return the whole string as a OSTRING */
			    goto String;
			}
		    }
			
		    iSeq = atoi(sHead);
		    cCont[2] = cContainerFindSequence( cCont[1], 
						DIRECTCONTENTS, iSeq );
		} else {
		    cCont[2] = 
			cContainerFindName( cCont[1], DIRECTCONTENTS, sHead );
		}
		return((OBJEKT)cCont[2]);
	    } else {
		return((OBJEKT)cCont[1]);
	    }
	} else {
	    goto String;
	}
    }


    if ( bDot ) {
	k = 0;
	do {
	    sRemoveLeadingSpaces( sLine );
	    sRemoveFirstString( sLine, sHead );
	    if ( strlen(sHead) == 0 ) break;
	    k++;
	    if ( cCont[k-1]->oHeader.cObjType == PARMSETid ) {
		/*  semi-HACK - parmsets don't have contents in this sense */
		cCont[k] = NULL;
	    } else if ( isdigit(sHead[0]) ) {

			    /* Make sure the rest are digits */
			    /* If not return NULL */
		for ( j=1; j<strlen(sHead); j++ ) {
		    if ( !isdigit(sHead[j]) ) {
   				/* It is not an object variable so...
			    	   return the whole string as a OSTRING */
			goto String;
    		    }
		}
		iSeq = atoi(sHead);
		cCont[k] = cContainerFindSequence( cCont[k-1], 
					    DIRECTCONTENTS, iSeq );
	    } else {
		cCont[k] = 
			cContainerFindName( cCont[k-1], DIRECTCONTENTS, sHead );
	    }
	    if ( cCont[k] == NULL ) break;
	} while ( strlen(sHead) != 0 ) ;
        if ( cCont[k] == NULL ) {
    		/* It is not an object variable so...
    		   return the whole string as a OSTRING */
		goto String;
    	}
	return((OBJEKT)cCont[k]);
    }

			/* If group notation then return the group */

    if ( bAt ) {
	sRemoveLeadingSpaces( sLine );
	sRemoveFirstString( sLine, sGroup );
	if ( iObjectType(cCont[0]) != UNITid ) {
    		/* It is not an object variable so...
    		   return the whole string as a OSTRING */
		goto String;
	}

	lGroup = lUnitGroup( (UNIT)cCont[0], sGroup );
	return((OBJEKT)lGroup);
    }

String:

   /* 
    *  It is not an object variable so...
    *	   return the whole string as a OSTRING
    *	   -- need to set refs to 0 so that
    *	      it will be freed later.. HACK
    */

    osString = (OSTRING)oCreate(OSTRINGid);
    OStringDefine( osString, sName );
    ((OBJEKT)(osString))->iReferences = 0;
    return((OBJEKT)osString );
}





    
/*
 *================================================================
 *
 *	Public routines
 */




/*
 *	ParseArguments
 *
 *	Parse the arguments that the user passes to LEaP from
 *	the command line arguments.
 */
void
ParseArguments( int argc, char *argv[] )
{
char		c;
extern	char	*optarg;

    while ( (c = getopt( argc, argv, "hsI:f:" )) != (char)(EOF) ) {
	switch (c) {
	    case 'h':
		printf( "Usage: %s [options]\n", argv[0] );
		printf( "Options:\n" );
		printf( " -h         Generate this message.\n" );
		printf( " -s         Ignore %s startup file.\n", LEAPRC );
		printf( " -I {dir}   Add {dir} to search path.\n" );
		printf( " -f {file}  Source {file}.\n" );
		exit(1);
	    case 's':
		printf( "-s: Ignoring startup file: %s\n", LEAPRC );
		SbUseStartup = FALSE;
		break;
	    case 'I':
		printf( "-I: Adding %s to search path.\n", optarg );
		BasicsAddDirectory( optarg, 1 );
		break;
	    case 'f':
		printf( "-f: Source %s.\n", optarg );
		if ( iFirstSource == 0 ) {
			MALLOC( SbFirstSourceFiles, STRING *, sizeof(STRING) );
			iFirstSource = 1;
		} else {
			iFirstSource++;
			REALLOC( SbFirstSourceFiles, STRING *, SbFirstSourceFiles,
					iFirstSource * sizeof(STRING));
		}
		strcpy( SbFirstSourceFiles[iFirstSource-1], optarg );
		break;
	}
    }
}






/*
 *	ParseInit
 *
 *	Initialize the parser.
 *	If SbStartup is TRUE the execute the LEAPRC script.
 */
void
ParseInit( RESULTt *rPResult )
{
FILE	*fStartup;
int	iFile;

    VP0(( "\nWelcome to LEaP!\n" ));

#ifdef  DEBUG
    VP0(( "LEaP is running in DEBUG mode!\n" ));
#endif

		/* Initialize memory manager debugging */
    INITMEMORYDEBUG();

    HelpInitialize();

                /* Initialize the first file in the stack to be stdin */
                
    GfaInputFileStack[0] = NULL;
    GbExecute = bBlockCreate();

                /* Create a few OBJEKTs that will be used by the parser */

    aDummy = (ATOM)oCreate(ATOMid);
    ContainerSetName( aDummy, "DUMMY" );
    GplAllParameters = plParmLibCreate();

    VariablesInit();
    GrMainResult.iCommand = CNONE;
    rPResult->iCommand = CNONE;
    
                /* Parse the LEAPRC file if bUseStartup is TRUE */

    if ( SbUseStartup ) {
        fStartup = FOPENNOCOMPLAIN( LEAPRC, "r" );
        if ( fStartup == NULL ) {
	    VP0(( "(no %s in search path)\n", LEAPRC ));
	} else {
	    /*
	     *  source the leaprc
	     */
	    VP0(( "Sourcing %s: %s\n", LEAPRC, GsBasicsFullName ));
	    INPUTPUSHFILE( fStartup );
	    GiClipPrompts = 1;
	    while ( fINPUTFILE() != NULL ) {
	        yyparse();
		if ( GrMainResult.iCommand == CQUIT ) {
	    	    if ( fINPUTFILE() != NULL )
	    	    	fclose( fINPUTFILE() );
	    	    INPUTPOPFILE();
		}
	    }
	    *rPResult = GrMainResult;
	    GiClipPrompts = 0;
	}
    }

		/* Parse the first source file specified */
		/* on the command line using the -f option */

    for (iFile=0; iFile<iFirstSource; iFile++) {
	fStartup = FOPENCOMPLAIN( SbFirstSourceFiles[iFile], "r" );
	if ( fStartup != NULL ) {
	    VP0(( "Sourcing: %s\n", GsBasicsFullName ));
	    INPUTPUSHFILE( fStartup );
	    GiClipPrompts = 1;
	    while ( fINPUTFILE() != NULL ) {
		yyparse();
		if ( GrMainResult.iCommand == CQUIT ) {
	    	    if ( fINPUTFILE() != NULL )
	    	    	fclose( fINPUTFILE() );
	    	    INPUTPOPFILE();
		}
	    }
	    *rPResult = GrMainResult;
	    GiClipPrompts = 0;
	}
    }
    if ( SbFirstSourceFiles )
	FREE( SbFirstSourceFiles );
}




/*
 *	ParseBlock
 *
 *	Parse a BLOCK containing one complete command.
 *	Return in rResult the result of the command.
 */
void
ParseBlock( BLOCK bBlock, RESULTt *rPResult )
{
		/* Set up the BLOCK from which to read the command */

    MESSAGE(( "Parsing block: %s\n", sBlockText(bBlock) ));
    GbCommand = bBlock;
    BlockResetRead( GbCommand );

    GrMainResult.iCommand = CNONE;

		/* Parse the BLOCK */
		/* Keep parsing as long as the 'execute' command */
		/* keeps setting the GLOBAL variable GbContinueParsing */

    do {
	yyparse();
	if ( GrMainResult.iCommand == CQUIT ) {
	    if ( fINPUTFILE() != NULL )
	    	fclose( fINPUTFILE() );
	    INPUTPOPFILE();
	}
    } while ( fINPUTFILE() != NULL );

	/* Reset the bCommandFound variable as a new command may be
	   available in a new input */
    bCommandFound = FALSE;
    
	/* Copy the RESULT from the global result variable */

    *rPResult = GrMainResult;

}





/*
 *	ParseShutdown
 *
 *	Shutdown the parser, release all variables setup in ParseInit
 *	Only needed if debugging memory mgt, since prog mem is all
 *	freed when process exits anyway.
 */
void
ParseShutdown()
{
	if ( !iMemDebug )
		return;

	Destroy( (OBJEKT *)&aDummy );
	VariablesDestroy();
	ParmLibDestroy( &GplAllParameters );

	BlockDestroy( &GbExecute );

	HelpShutdown();
}
